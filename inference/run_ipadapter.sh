export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=2 python inf_ip_sft.py \
    --model_path "../finetune/CogVideoX-2b" \
    --dtype "bfloat16" \
    --output_dir "/private/task/huangmq/CogVideo/finetune/results/Cogvideox-2b-adapter-5e-4/epochs_1" \
    --output_mp4_file "train_image.mp4" \
    --generate_type "t2v" \
    --prompt "A dog is the focal point against a gray backdrop. Initially, the dog's tongue is playfully out, and its ears are perked up, adding to its charming demeanor. As time passes, the dog's expressions and poses remain consistent, with the tongue occasionally protruding, suggesting a playful and whimsical atmosphere. The dog's attire and the plain background emphasize its unique style and the humorous contrast between its cool demeanor and the warm, inviting setting. The dog's alert and curious gaze occasionally shifts, maintaining the playful and whimsical mood." \
    --ipadapter_path "/private/task/huangmq/CogVideo/finetune/outputs/CogVideoX-2b/ipadapter_4e-5/checkpoint-400" \
    --num_frames 49 \
    --image_or_video_path '/private/task/tuyijing/Datasets/dog_dataset_eval/ref_images/2/2_masked.jpg' \
    --width 720 \
    --height 480 \

    # --prompt "A dog is seen lying on its stomach in various states of relaxation and alertness on a grassy lawn. Its coat contrasts with the green grass, and its ears are perked up, suggesting curiosity. Its eyes are alert and expressive, sometimes gazing directly at the camera, other times closing in contentment. The dog's posture changes subtly, with its front paws stretched out and its head resting on its forelimbs, indicating moments of rest and playfulness. The serene outdoor setting is highlighted by soft natural lighting, emphasizing the dog's elegant form and the tranquil mood."
    # --prompt "In the warm sunshine, a dog with a playful expression and a lively tail runs through a field of tall grasses and colorful flowers, including pink and purple blooms. As it runs, its ears are perked up, and its eyes sparkle with joyful light. The background is dark tree trunks, adding depth to the scene. The dog's energetic movements and serene natural setting evoke a longing for the simple joys of youthful innocence" 
    # --prompt "A playful dog with shiny fur, energetically running across a lush green meadow under a bright, clear blue sky. The dog has a happy expression with its tongue slightly out, and its tail wagging rapidly. The grass sways gently in the breeze, dotted with colorful wildflowers. Sunlight bathes the entire scene, casting soft shadows and creating a lively, vibrant atmosphere. The dog 's paws kick up small bits of grass and dust as it sprints, conveying a sense of motion and joy." \

    # --image_or_video_path '/private/task/tuyijing/CogVideo/sat/dog_inference_test/image0.jpg' \
    # --image_or_video_path '/private/task/tuyijing/CogVideo/sat/dog_inference_test/image1_masked.png' \
    # --image_or_video_path '/private/task/tuyijing/CogVideo/sat/dog_inference_test/image2_masked.jpg' \