CUDA_VISIBLE_DEVICES=4 python main.py --simplified --eta 1.0 --config imagenet_256.yml --dataset div2k --deg_scale 4.0 --alpha 0.3 --total_step 100 \
    --mode implicit --DDNM_A implicit --DDNM_Ap implicit --posterior_formula DDIM --save_img \
    --perturb_y --perturb_A implicit --perturb_Ap implicit \
    --Learning_degradation --IRopt ./configs/Test2e_div2k_blind_x4.yaml \
    --image_folder "test/div2k_x4" --path_y /eva_data4/shlu2240/Boosting-Diffusion-Guidance-via-Learning-Degradation-Aware-Models-for-Blind-Super-Resolution/input/DIV2K
