<div align="center">
<h1>MAS-SAM(IJCAI2024)</h1>
<h3>MAS-SAM: Segment Any Marine Animal with Aggregated Features</h3>

[Tianyu Yan](https://github.com/Drchip61)<sup>1</sup>,[Zifu Wan](zifuw@andrew.cmu.edu)<sup>2</sup>,[Xinhao Deng](dengxh@mail.dlut.edu.cn)<sup>1</sup>,[*Pingping Zhang✉️](http://faculty.dlut.edu.cn/zhangpingping/en/index.htm)<sup>1</sup>,[Yang Liu](http://faculty.dlut.edu.cn/liuyang1/zh_CN/index.htm)<sup>1</sup>, [Huchuan Lu](http://faculty.dlut.edu.cn/Huchuan_Lu/zh_CN/index.htm)<sup>1</sup>

[Dalian University of Technology, IIAU-Lab](https://futureschool.dlut.edu.cn/IIAU.htm)<sup>1</sup>
Robotics Institute, Carnegie Mellon University<sup>2</sup>


## Abstract

Recently, Segment Anything Model (SAM) shows exceptional performance in generating high-quality object masks and achieving zero-shot image segmentation. However, as a versatile vision model, SAM is primarily trained with large-scale natural light images. In underwater scenes, it exhibits substantial performance degradation due to the light scattering and absorption. Meanwhile, the simplicity of the SAM’s decoder might lead to the loss of fine-grained object details. To address the above issues, we propose a novel feature learning framework named MAS-SAM for marine animal segmentation, which involves integrating effective adapters into the SAM’s encoder and constructing a pyramidal decoder. More specifically, we first build a new SAM’s encoder with effective adapters for underwater scenes. Then, we introduce a Hypermap Extraction Module (HEM) to generate multi-scale features for a comprehensive guidance. Finally, we propose a Progressive Prediction Decoder (PPD) to aggregate the multi-scale features and predict the final segmentation results. When grafting with the Fusion Attention Module (FAM), our method enables to extract richer marine information from global contextual cues to fine-grained local details. Extensive experiments on four public MAS datasets demonstrate that our MAS-SAM can obtain better results than other typical segmentation methods.



## Getting Started

</div>

### Installation

**step1:Clone the Dual_SAM repository:**

To get started, first clone the Dual_SAM repository and navigate to the project directory:

```bash
git clone https://github.com/Drchip61/Dual_SAM.git
cd Dual_SAM

```

**step2:Environment Setup:**

Dual_SAM recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:
#### Create and activate a new conda environment

```bash
conda create -n Dual_SAM
conda activate Dual_SAM
```
#### Install Dependencies.
```bash
pip install -r requirements.txt
```

#### Download pretrained model.
Please put the pretrained [SAM model](https://drive.google.com/file/d/1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg/view?usp=share_link) in the Dual-SAM file.

### Model Training and Testing

**Training**
```bash
# Change the hyper parameter in the train_s.py 
python train_s.py
```

**Testing**
```bash
# Change the hyper parameter in the test_y.py 
python test_y.py
```

### Analysis Tools


```bash
# First threshold the prediction mask
python bimap.py
# Then evaluate the perdiction mask
python test_score.py
```

## Citation

```
@inproceedings{
anonymous2024fantastic,
title={Fantastic Animals and Where to Find Them: Segment Any Marine Animal with Dual {SAM}},
author={Pingping Zhang，Tianyu Yan， Yang Liu，Huchuan Lu},
booktitle={Conference on Computer Vision and Pattern Recognition 2024},
year={2024}
}
```
