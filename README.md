# Boosting Diffusion Guidance via Learning Degradation Aware Models for Blind Super Resolution (WACV2025 Oral)

[Shao-Hao Lu](https://www.linkedin.com/in/shao-hao-lu-b629692a8/), [Ren Wang](https://renwang0508.github.io/), [Ching-Chun Huang](https://nycu-acm.github.io/ACM_NYCU_website/members/Ching-Chun-Huang.html), [Wei-Chen Chiu](https://walonchiu.github.io/)


[Paper](https://arxiv.org/abs/2501.08819) | [Supplementary](assets/0265_supp.pdf) | [Video](https://www.youtube.com/watch?v=ZOUJ0QiSnRc&feature=youtu.be)

:tada: Accepted to **WACV'25** Algorithm Track :tada:

This is the **official repository** of the [**paper**](https://arxiv.org/abs/2501.08819) "Boosting Diffusion Guidance via Learning Degradation Aware Models for Blind Super Resolution".



## Overview
>Recently, diffusion-based blind super-resolution (SR) methods have shown great ability to generate high-resolution images with abundant high-frequency detail, but the detail is often achieved at the expense of fidelity. Meanwhile, another line of research focusing on rectifying the reverse process of diffusion models (i.e., diffusion guidance), has demonstrated the power to generate high-fidelity results for non-blind SR. However, these methods rely on known degradation kernels, making them difficult to apply to blind SR. To address these issues, we present DADiff in this paper. DADiff incorporates degradation-aware models into the diffusion guidance framework, eliminating the need to know degradation kernels. Additionally, we propose two novel techniques---input perturbation and guidance scalar---to further improve our performance. Extensive experimental results show that our proposed method has superior performance over state-of-the-art methods on blind SR benchmarks.
><img src="./assets/framework.png" align="middle" width="800">

## Evaluation
### Qualitative comparision
<p align="center">
<img src="./assets/div2k.png" align="middle" width="800">
</p>

Qualitative comparison of 4× upsampling on DIV2K-Val. The magnified areas are indicated with red boxes.

<p align="center">
<img src="./assets/celeba.png" align="middle" width="800">
</p>

Qualitative comparison of 4× upsampling on CelebA-Val.

### Quantitative comparison
<p align="center">
<img src="./assets/quan.png" align="middle" width="800">
</p>

Evaluation and experimental results demonstrate that
1. Compared to DDNM, our method outperforms DDNM in both fidelity and perceptual quality, showing the effectiveness of our degradation-aware models.
2. Comparing our method to diffusion-based blind super-resolution methods, we also excel in both fidelity and perceptual quality, successfully addressing the challenge of the fidelity weakness often associated with generative-based blind-SR methods.
3. compared to MsdiNet, which is exactly our restoration model Gr, our method demonstrates superior perceptual quality, but at the expense of fidelity because MsdiNet is a regression-based method.



## Installation
### Code
```
git clone https://github.com/ryanlu2240/DADiff.git
```
### Environment
```
conda env create -f environment.yml
conda activate DADiff
```
### Pre-Trained Models
To restore human face images, download this [model](https://drive.google.com/file/d/1_V4wIVciyPayyzBs3H5IchegAt9W8Jks/view?usp=sharing) (from [SDEdit](https://github.com/ermongroup/SDEdit)). 
```
https://drive.google.com/file/d/1_V4wIVciyPayyzBs3H5IchegAt9W8Jks/view?usp=sharing
```
To restore general images, download this [model](https://drive.google.com/file/d/1ToiNqyDnxj9r6hrpd5J43QuBC7G1NP4O/view?usp=sharing) (from [guided-diffusion](https://github.com/openai/guided-diffusion)).
```
https://drive.google.com/file/d/1ToiNqyDnxj9r6hrpd5J43QuBC7G1NP4O/view?usp=sharing
```
Download degradation-aware models [celeba_x4](https://drive.google.com/file/d/1WF_cvSrIY7ltSYQoqK5JK1zvAyAhBVwX/view?usp=sharing), [celeba_x8](https://drive.google.com/file/d/13oUxCWDKLRLDu7O1kci4WaW7e2y-WX6w/view?usp=sharing), [div2k_x4](https://drive.google.com/file/d/1aKVfQlx6MbgLtJarTg4eBv9SlAhiXssJ/view?usp=sharing). 
### Setting
The detailed sampling command is here:
```
python main.py --simplified --eta {ETA} --config {diffusion_CONFIG} --dataset celeba --deg_scale {DEGRADATION_SCALE} --alpha {GUIDANCE_SCALAR} --total_step 100 --mode implicit --DDNM_A implicit --DDNM_Ap implicit --posterior_formula DDIM --perturb_y --perturb_A implicit --perturb_Ap implicit --Learning_degradation --IRopt {Degradation-Aware_CONFIG} --image_folder {IMAGE_FOLDER} --path_y {INPUT_PATH} --diffusion_ckpt {DIFFUSION_CKPT} --save_img
```
with following options:
- `INPUT_PATH` is the root folder of input image.
- `ETA` is the DDIM hyperparameter. 
- `DEGRADATION_SCALE` is the scale of degredation.
- `diffusion_CONFIG` is the name of the diffusion model config file.
- `GUIDANCE_SCALAR` is the proposed guidance scalar.
- `DIFFUSION_CKPT` is the path of pretrain diffusion checkpoint.
- `IMAGE_FOLDER` is the folder name of the results.
- `Degradation-Aware_CONFIG` is the MsdiNet config file, include the degradation-aware model checkpoint setting.


### Quick Start
Run below command to get 4x SR results immediately. The results should be in `DADiff/exp/result/`.
```
bash run_celeba_x4.sh
```

## Reproduce The Results In The Paper
### Quantitative Evaluation
Testing Dataset download link: [Google drive](https://drive.google.com/file/d/12t60HfSwosHxZMhk_aulveMfp79Gg7QQ/view?usp=sharing)

Our testing result: [Google drive](https://drive.google.com/file/d/1doevxFAxqQfmlY5Y2vIIJFMVh5Cb5RuI/view?usp=sharing)


## References
If you find this repository useful for your research, please cite the following work.
```
@article{lu2025boosting,
  title={Boosting Diffusion Guidance via Learning Degradation-Aware Models for Blind Super Resolution},
  author={Lu, Shao-Hao and Wang, Ren and Huang, Ching-Chun and Chiu, Wei-Chen},
  journal={arXiv preprint arXiv:2501.08819},
  year={2025}
}
```
## Acknowledgement

This project is built upon the the gaint sholder of [DDNM](https://github.com/wyhuai/DDNM/) and [MsdiNet](https://github.com/dasongli1/Learning_degradation). Great thanks to them!


