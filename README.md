# NLAQNet

# Non-Local Feature Aggregation Quaternion Network for Single Image Deraining

## Abstract

> Since outdoor visual tasks would be affected by the image degradation due to rainmarks on rainy days, it makes sense to learn image rain removal tasks. The existing mainstream denoising methods use convolutional neural networks(CNN) to learn the mapping relationship between rainy and clean images. However, the real-valued CNN process the color images as three independent channels separately, which fails to fully leverage color information. Additionally, sliding-window-based neural networks cannot effectively model the non-local characteristics of an image. To address these issues, we propose a non-local feature aggregation quaternion network (NLAQNet), which is composed of two concurrent sub-networks: the Quaternion Local Detail Repair Network (QLDRNet) and the Multi-Level Feature Aggregation Network (MLFANet). Furthermore, in the subnetwork of QLDRNet, the Quaternion Residual Attention Block (QRSB) and Local Detail Repair Block (LDRB) are proposed to repair the backdrop of an image that has not been damaged by rain streaks. Finally, in the subnetwork of MLFANet, we additionally designed Real Residual Self-Attention Block (RSB), Non-Local Feature Aggregation Block (NLAB), and Feature Aggregation Block (Mix), which are targeted towards repairing the backdrop of an image that has been damaged by rain streaks. Extensive experiments demonstrate that the proposed network outperforms state-of-the-art single image deraining networks in both qualitative and quantitative comparisons on existing datasets.


![NLAQ5](https://github.com/xionggonghe/NLAQNet/blob/master/images/NLAQ.jpg)

## Prepare

```

install toch 1.11
install numpy 1.23.3
install opencv-python 4.6.0.66
install lpips 0.1.3
install tqdm 4.64.0
install torchvision 0.12.0
install tensorboardX 1.2
```

Download the dataset from (Link：https://pan.baidu.com/s/1HomXxQISUJcER8bwsGrdgA ,Password： kex6)  put the dataset folder into the "NLAQNet/data" folder

## Training

```
 python train.py --train_data ./data/mixtrain/ --val_data ./data/mixtest/Rain100H/  --save_weights ./pretrain_model/ --batchSize 7 --pachSize 128 --loadWeights False
```

## Testing

```
python test.py --read_weights ./pretrain_model/model_best_Mixed.pth --test_dataset Rain100H --cropPatch False

Test results can be obtained from this  (Link：https://pan.baidu.com/s/1HomXxQISUJcER8bwsGrdgA ,Password： kex6)

We will upload the trained model after the paper is accepted.
```

**To reproduce PSNR/SSIM scores of the paper, run**

```
evaluate_PSNR_SSIM.m 
```



## Citation

