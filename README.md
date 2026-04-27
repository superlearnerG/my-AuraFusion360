<p align="center">

  <h2 align="center">AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting </h2>
  <p align="center">
    <a href="https://kkennethwu.github.io/"><strong>Chung-Ho Wu*</strong></a>
    ·
    <a href=""><strong>Yang-Jung Chen*</strong></a>
    ·
    <a href=""><strong> Ying-Huan Chen</strong></a>
    ·
    <a href="https://jayinnn.dev/"><strong>Jie-Ying Lee</strong></a>
    ·
    <a href="https://hentci.github.io/"><strong>Bo-Hsu Ke</strong></a>
    ·
    <a href=""><strong>Chun-Wei Tuan Mu</strong></a>
    ·
    <a href=""><strong> Yi-Chuan Huang</strong></a>
    ·
    <a href="https://linjohnss.github.io/"><strong>Chin-Yang Lin</strong></a>
    ·
    <a href="https://minhungchen.netlify.app/"><strong>Min-Hung Chen</strong></a>
    ·
    <a href="https://sites.google.com/site/yylinweb/"><strong>Yen-Yu Lin</strong></a>
    <br>
    ·
    <a href="https://yulunalexliu.github.io/"><strong>Yu-Lun Liu</strong></a>
    <br>
    <br>
        <a href="http://arxiv.org/abs/2502.05176"><img src='https://img.shields.io/badge/arXiv-2502.05176-red' alt='Paper PDF'></a>
        <a href='https://kkennethwu.github.io/aurafusion360/'><img src='https://img.shields.io/badge/Project_Page-AuraFusion360-green' alt='Project Page'></a>
        <a href='https://drive.google.com/drive/folders/1C0OqUSavUBwR_p_yNRBia90jvC-23hNN?usp=sharing'><img src='https://img.shields.io/badge/Dataset-360USID-blue' alt='Project Page'></a>
        <a href='https://huggingface.co/datasets/kkennethwu/360-USID'><img src='https://img.shields.io/badge/Dataset(HF)-360USID-blue' alt='Project Page'></a>
        <a href='https://drive.google.com/drive/folders/1ujTKiBfTBD_BVv9YiifOl_SfG8sd0dZ6'><img src='https://img.shields.io/badge/Evaluation Results-AuraFusion360-orange' alt='Project Page'></a>
    <br>
    <b> NYCU |&nbsp;NVIDIA </b>
  </p>

  <table align="center">
    <tr>
    <td>
      <img src="assets/Figures/teaser.png">
    </td>
    </tr>
  </table>
<p>

## News
* **[2025.02.10]** <img src="assets/Figures/favicon.svg" alt="icon" style="height: 1em; vertical-align: -0.5mm;"> Release project page, arXiv paper, dataset, and evaluation results!
* **[2025.02.27]** Accepted by CVPR 2025!
* **[2025.06.29]** Release Full Code.
* **[2025.06.30]** All AuraFusion360 results are now available on [HuggingFace](https://huggingface.co/datasets/kkennethwu/AuraFusion360_Results). Note that these results may differ slightly from the paper. For the complete evaluation results and other baseline methods presented in the paper, please refer to [here](https://drive.google.com/drive/folders/1ujTKiBfTBD_BVv9YiifOl_SfG8sd0dZ6).

## Get Started
### Environment Setup
```
git clone https://github.com/kkennethwu/AuraFusion360_official.git --recursive
export HF_TOKEN=<your hf token>
export HF_HOME=<your hf home>
source install.sh
```

### Download Dataset
In addition to Google Drive, the 360-USID (our dataset) and Other-360 (collected dataset) are now available for download via HuggingFace.
```
huggingface-cli login
huggingface-cli download kkennethwu/360-USID --repo-type dataset --local-dir ./data --resume-download --quiet --max-workers 32
```

### Running
#### 1. Training Object-Masked Gaussians
```
python train.py --config configs/{dataset_name}/{scene_name}/train.config
python render.py -s data/{dataset_name}/{scene_name} -m output/{dataset_name}/{scene_name} --skip_mesh --render_path --iteration 30000
```
#### 2. Removing Objects & Generating Unseen Masks
```
python remove.py --config configs/{dataset_name}/{scene_name}/remove.config
python utils/sam2_utils.py --dataset {dataset_name} --scene {scene_name}
# python scripts/visualize_mask.py --dataset {dataset_name} --scene {scene_name} --type mask # (optional) 
# python scripts/visualize_mask.py --dataset {dataset_name} --scene {scene_name} --type contour # (optional)
```
#### 3. Unproject & Inpaint
```
python inpaint.py --config configs/$dataset_name/$scene_name/inpaint.config
python utils/LeftRefill/sdedit_utils.py --config configs/$dataset_name/$scene_name/sdedit.config 
python inpaint.py --config configs/$dataset_name/$scene_name/inpaint.config --images inpaint --finetune_iteration 10000
```


## Citation
If you find our dataset, evaluation results, or code useful, please cite this paper and give us a ⭐️.
```BibTex
@InProceedings{wu2025aurafusion,
    author    = {Wu, Chung-Ho and Chen, Yang-Jung and Chen, Ying-Huan and Lee, Jie-Ying and Ke, Bo-Hsu and Mu, Chun-Wei Tuan and Huang, Yi-Chuan and Lin, Chin-Yang and Chen, Min-Hung and Lin, Yen-Yu and Liu, Yu-Lun},
    title     = {AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360deg Unbounded Scene Inpainting},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {16366-16376}
}
```

## Acknowledgements
This work was supported by NVIDIA Taiwan AI Research & Development Center (TRDC).
This research was funded by the National Science and Technology Council, Taiwan, 
under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-E-A49-023-. 
Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.
