<div align="center">

<!-- <h1 align="center">
  <span style="background: linear-gradient(90deg, #FFB74D, #4CAF50, #2196F3); -webkit-background-clip: text; color: transparent; display: inline-block;">PG-Occ</span>
</h1> -->

<img src='./assets/title.png'></img>

[Chi Yan](https://yanchi-3dv.github.io)<sup>1,2</sup>, [Dan Xu](https://www.danxurgb.net/)<sup>1</sup><br>
<sup>1</sup>The Hong Kong University of Science and Technology, <sup>2</sup>ZEEKR

<a href='https://arxiv.org/abs/2510.04759'><img src='https://img.shields.io/badge/ArXiv-2510.04759-red'></a> &nbsp; 
<a href='https://yanchi-3dv.github.io/PG-Occ/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  &nbsp;
</div>

<img src='./assets/head.png'></img>

Demo videos are available at the [project page](https://yanchi-3dv.github.io/PG-Occ/).

## 📌 TODO
We are organizing the code and applying to the company, please be patient
- [x] Release project page
- [x] Release checkpoint
- [x] Release training code

## 🔆 Abstract

> The 3D occupancy prediction task has witnessed remarkable progress in recent years, playing a crucial role in vision-based autonomous driving systems. While traditional methods are limited to fixed semantic categories, recent approaches have moved towards predicting text-aligned features to enable open-vocabulary text queries in real-world scenes. However, there exists a trade-off in text-aligned scene modeling: sparse Gaussian representation struggles to capture small objects in the scene, while dense representation incurs significant computational overhead. To address these limitations, we present PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables open-vocabulary 3D occupancy prediction. Our framework employs progressive online densification, a feed-forward strategy that gradually enhances the 3D Gaussian representation to capture fine-grained scene details. By iteratively enhancing the representation, the framework achieves increasingly precise and detailed scene understanding. Another key contribution is the introduction of an anisotropy-aware sampling strategy with spatio-temporal fusion, which adaptively assigns receptive fields to Gaussians at different scales and stages, enabling more effective feature aggregation and richer scene information capture. Through extensive evaluations, we demonstrate that PG-Occ achieves state-of-the-art performance with a relative 14.3% mIoU improvement over the previous best performing method.

## 😉 Pipline
![first_fig4_00](assets/pipeline.png)

## 📦 Environment

Create conda environment:

```bash
conda create -n pgocc python=3.8
conda activate pgocc
pip install -r requirements.txt
```

Install turbojpeg and pillow-simd to speed up data loading (optional but important):

```bash
sudo apt-get update
sudo apt-get install -y libturbojpeg
pip install pyturbojpeg
```

Install other dependencies:
> **Note:** If you encounter any issues during installation, we recommend building from source code.

```bash
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
```

Compile CUDA extensions:

```bash
cd models/csrc
python setup.py build_ext --inplace
cd ../../

cd lib/pointops
pip install .
cd ../../
```

## 📂 Prepare Dataset
1. Download nuScenes from [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes), put it to `data/nuscenes`.
2. Download the generated info file from [gdrive](https://drive.google.com/drive/folders/1OTqpRHgqXFynBKYiqDk0lkQc8QMt_I2a?usp=sharing).
3. Download gts-nuScenes occupancy GT from [gdrive](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link), unzip it, and save it to `data/nuscenes/gts`.
4. Generate MaskCLIP features and Depth rendering targets step are same as [GaussTR](https://github.com/hustvl/GaussTR), and save it to `data/nuscenes_featup` and `data/nuscenes_metric3d`.
5. (Optional) Generate the ground truth depth maps for validation following [GaussianOcc](https://github.com/GANWANSHUI/GaussianOcc).

## 🚀 Training & Evaluation

Train PG-Occ with multi GPUs:
```bash
bash scripts/train.sh [num_gpus] [resume_from]
```
For example:
```bash
bash scripts/train.sh 8
```

Evaluate PG-Occ:

> **Note:** Download the [checkpoint](https://github.com/yanchi-3dv/PG-Occ/releases) and save it to `ckpt/pg_occ_maskclip_8_miou15.15.pth`.

```bash
bash scripts/val.sh [num_gpus] [weights]
```

For example:

```bash
bash scripts/val.sh 1 ckpt/pg_occ_maskclip_8_miou15.15.pth
```

## 📭 Citation

If you find PG-Occ helpful to your research, please cite our paper:
```
@article{yan2025pgocc,
  title={Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction},
  author={Yan, Chi and Xu, Dan},
  journal={arXiv preprint arXiv:2510.04759},
  year={2025}
}
```

## 🙏 Acknowledgements

Many thanks to these excellent open-source projects: [SparseBEV](https://github.com/MCG-NJU/SparseBEV), [GaussTR](https://github.com/hustvl/GaussTR), [gsplat](https://github.com/nerfstudio-project/gsplat), [LangOcc](https://github.com/boschresearch/LangOcc), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [GaussianFormer](https://github.com/huang-yh/GaussianFormer), [GaussianOcc](https://github.com/GANWANSHUI/GaussianOcc), [4D-Occ](https://github.com/tarashakhurana/4d-occ-forecasting), and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

