from diffusers.configuration_utils import ConfigMixin,register_to_config
import torch
import torch.nn as nn
import math
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.transformers import CogVideoXTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND,is_torch_version,logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import get_2d_sincos_pos_embed,TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
import copy
import torch.nn.functional as F

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class GeneralIP_AttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class AttnBlock(nn.Module):
    def __init__(self, 
                attn1,
                is_training:bool,
                dim: int,
                num_attention_heads: int,
                attention_head_dim: int,
                ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if is_training:
            self.attn = copy.deepcopy(attn1)
            self.attn.set_processor(GeneralIP_AttnProcessor())
        else:
            self.attn = Attention(
                    query_dim=dim,
                    dim_head=attention_head_dim,
                    heads=num_attention_heads,
                    qk_norm="layer_norm",
                    eps=1e-6,
                    bias=True,
                    out_bias=True,
                    processor=GeneralIP_AttnProcessor(),
                )

    def forward(self, ref_image_hidden_states):
        norm_ref_image = self.norm(ref_image_hidden_states)
        attn_ref_image = self.attn(norm_ref_image)
        return attn_ref_image

@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        ref_image_bias: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # insert here
        # hidden_states = hidden_states + gate_msa * attn_hidden_states

        hidden_states = hidden_states + gate_msa * attn_hidden_states + ref_image_bias
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

class GeneralIP_CogVideoXTransformer3DModel(CogVideoXTransformer3DModel):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        ref_image_dim: int = 768,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
    ):
        super().__init__(
            num_attention_heads = num_attention_heads,
            attention_head_dim = attention_head_dim,
            in_channels = in_channels,
            out_channels =  out_channels,
            flip_sin_to_cos = flip_sin_to_cos,
            freq_shift = freq_shift,
            time_embed_dim = time_embed_dim,
            text_embed_dim = text_embed_dim,
            num_layers = num_layers,
            dropout = dropout,
            attention_bias = attention_bias,
            sample_width = sample_width,
            sample_height = sample_height,
            sample_frames = sample_frames,
            patch_size = patch_size,
            temporal_compression_ratio = temporal_compression_ratio,
            max_text_seq_length = max_text_seq_length,
            activation_fn = activation_fn,
            timestep_activation_fn = timestep_activation_fn,
            norm_elementwise_affine = norm_elementwise_affine,
            norm_eps = norm_eps,
            spatial_interpolation_scale = spatial_interpolation_scale,
            temporal_interpolation_scale = temporal_interpolation_scale,
            use_rotary_positional_embeddings = use_rotary_positional_embeddings,
            use_learned_positional_embeddings = use_learned_positional_embeddings,
            )
        self.inner_dim = attention_head_dim * num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.ref_image_dim = ref_image_dim
        self.num_layers= num_layers

        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def add_blocks_for_ref(self,is_training):
        self.blocks_for_ref = nn.ModuleList([
            AttnBlock(
                    attn1=self.transformer_blocks[i].attn1,
                    is_training=is_training,
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    ) for i in range(self.num_layers)
                ])
        
        self.map = nn.Linear(self.ref_image_dim, self.inner_dim)


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        ref_image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape # [b,13,16,60,90]
        p = self.config.patch_size # 2

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length] #[b,226,1920]
        hidden_states = hidden_states[:, text_seq_length:] #[b,13*30*45,1920]

        ref_image_hidden_states = self.map(ref_image_hidden_states) #[b,1,1920]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                # weight_is_equal = torch.allclose(self.blocks_for_ref[i].attn.to_q.weight.to(torch.float32),block.attn1.to_q.weight.to(torch.float32))
                # bias_is_equal = torch.allclose(self.blocks_for_ref[i].attn.to_q.bias.to(torch.float32),block.attn1.to_q.bias.to(torch.float32))
                # print(f"block {i} to_q weight_is_close:{weight_is_equal}")
                # print(f"block {i} to_q bias_is_close:{bias_is_equal}")



                ref_image_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.blocks_for_ref[i]),
                    ref_image_hidden_states,
                    **ckpt_kwargs,
                )
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    ref_image_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                ref_image_hidden_states = self.blocks_for_ref[i](
                    ref_image_hidden_states=ref_image_hidden_states,
                )
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    ref_image_bias=ref_image_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify


        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


    


