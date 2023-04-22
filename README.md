# NLAQNet

# Non-Local Feature Aggregation Quaternion Network for Single Image Deraining

## Abstract

> Image rain removal is a highly ill-posed and complex computer vision task. Removing the influence of adverse weather conditions can enhance the quality of subsequent vision tasks. Existing methods mainly used convolutional neural networks to restore image backgrounds contaminated by rain streaks. However, widely used real-valued convolutional neural networks process color images as three independent channels, failing to fully leverage color information. Additionally, sliding-window-based neural networks cannot effectively model the non-local characteristics of an image


![NLAQ5](C:\Users\Xiong\OneDrive\Documents\研究生资料\论文书写\实验图表\PNG\NLAQ5.png)

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

Download the dataset from (Link：https://pan.baidu.com/s/1HomXxQISUJcER8bwsGrdgAPassword： kex6)  put the dataset folder into the "NLAQNet/data" folder

## Training

```
 python train.py --train_data ./data/mixtrain/ --val_data ./data/mixtest/Rain100H/  --save_weights ./pretrain_model/ --batchSize 7 --pachSize 128 --loadWeights False
```

## Testing

```
python test.py --read_weights ./pretrain_model/model_best_Mixed.pth --test_dataset Rain100H --cropPatch False

Test results can be obtained from this  (Link：https://pan.baidu.com/s/1HomXxQISUJcER8bwsGrdgAPassword： kex6)

We will upload the trained model after the paper is accepted.
```

**To reproduce PSNR/SSIM scores of the paper, run**

```
evaluate_PSNR_SSIM.m 
```



## Citation

