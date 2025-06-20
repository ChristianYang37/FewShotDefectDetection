# Few-Shot Learning Method for Customized Defect Detection

This repo is built based on [LibFewShot](https://github.com/RL-VIG/LibFewShot) and [FSOD](https://github.com/fanq15/FSOD-code/tree/master).

## Supported Methods

### Non-episodic methods (a.k.a Fine-tuning based methods)

+ [Baseline (ICLR 2019)](https://arxiv.org/abs/1904.04232)
+ [Baseline++ (ICLR 2019)](https://arxiv.org/abs/1904.04232)
+ [RFS (ECCV 2020)](https://arxiv.org/abs/2003.11539)
+ [SKD (BMVC 2021)](https://arxiv.org/abs/2006.09785)
+ [Negcos (ECCV 2020)](https://arxiv.org/abs/2003.12060)
+ [S2M2 (WACV 2020)](https://arxiv.org/abs/1907.12087)
+ [Meta-Baseline (ICCV 2021)](https://arxiv.org/abs/2003.04390)
+ [Diffkendall(NeurIPS 2023)](https://arxiv.org/abs/2307.15317)

### Meta-learning based methods

+ [MatchingNet (NeurIPS 2016)](https://arxiv.org/abs/1606.04080)
+ [MAML (ICML 2017)](https://arxiv.org/abs/1703.03400)
+ [Versa (NeurIPS 2018)](https://openreview.net/forum?id=HkxStoC5F7)
+ [R2D2 (ICLR 2019)](https://arxiv.org/abs/1805.08136)
+ [LEO (ICLR 2019)](https://arxiv.org/abs/1807.05960)
+ [MTL (CVPR 2019)](https://arxiv.org/abs/1812.02391)
+ [ANIL (ICLR 2020)](https://arxiv.org/abs/1909.09157)
+ [IFSL(NeurIPS 2020)](https://arxiv.org/abs/2009.13000)
+ [BOIL (ICLR 2021)](https://arxiv.org/abs/2008.08882)
+ [MeTAL (ICCV 2021)](https://arxiv.org/abs/2110.03909)

### Metric-learning based methods

+ [ProtoNet (NeurIPS 2017)](https://arxiv.org/abs/1703.05175)
+ [RelationNet (CVPR 2018)](https://arxiv.org/abs/1711.06025)
+ [ConvaMNet (AAAI 2019)](https://ojs.aaai.org//index.php/AAAI/article/view/4885)
+ [DN4 (CVPR 2019)](https://arxiv.org/abs/1903.12290)
+ [CAN (NeurIPS 2019)](https://arxiv.org/abs/1910.07677)
+ [ATL-Net (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/0100.pdf)
+ [ADM (IJCAI 2020)](https://arxiv.org/abs/2002.00153)
+ [DSN (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Simon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.pdf)
+ [FEAT (CVPR 2020)](http://arxiv.org/abs/1812.03664)
+ [RENet (ICCV 2021)](https://arxiv.org/abs/2108.09666)
+ [FRN (CVPR 2021)](https://arxiv.org/abs/2012.01506)
+ [DeepBDC (CVPR 2022)](https://arxiv.org/abs/2204.04567)
+ [MCL (CVPR 2022)](http://openaccess.thecvf.com/content/CVPR2022/html/Liu_Learning_To_Affiliate_Mutual_Centralized_Learning_for_Few-Shot_Classification_CVPR_2022_paper.html)
+ [CPEA (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Hao_Class-Aware_Patch_Embedding_Adaptation_for_Few-Shot_Image_Classification_ICCV_2023_paper.pdf)
+ FSOD + ProtoNet (Our customized method)

## Quick Installation

Please refer to [install.md](https://libfewshot-en.readthedocs.io/en/latest/install.html)([安装](https://libfewshot-en.readthedocs.io/zh_CN/latest/install.html)) for installation.

Complete tutorials can be found at [document](https://libfewshot-en.readthedocs.io/en/latest/)([中文文档](https://libfewshot-en.readthedocs.io/zh_CN/latest/index.html)).

## Current Results

We apply the multi-relation method from FSOD to ProtoNet; the code is located in `core/model/metric/fsod_net.py`. We evaluate ProtoNet and our method on the CIFAR-100 dataset for comparison. The training log, including the best accuracy of each method, is given in `./results`. The learning rate of our method is specially set to 0.001 since the training loss instability of the utilization with the default learning rate (0.01).

## ToDo List

* MML
* Meta Learning (MAML, Reptile)

## Available Datasets

[Caltech-UCSD Birds-200-2011](https://data.caltech.edu/records/20098), [Standford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [Standford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html), [*mini*ImageNet](https://arxiv.org/abs/1606.04080v2), [*tiered*ImageNet](https://arxiv.org/abs/1803.00676) and [WebCaricature](https://arxiv.org/abs/1703.03230) are available at [Google Drive](https://drive.google.com/drive/u/1/folders/1SEoARH5rADckI-_gZSQRkLclrunL-yb0) and [百度网盘(提取码：yr1w)](https://pan.baidu.com/s/1M3jFo2OI5GTOpytxgtO1qA).

## License

This project is licensed under the MIT License. See LICENSE for more details.
