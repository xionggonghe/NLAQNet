from model.networks import QDerainNet
import torch
from utils.PSNR import torchPSNR as PSNR
from utils.SSIM import SSIM
from utils.dataset_RGB import DataLoader_Test
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage
import numpy as np
import torchvision
import cv2
import time
import lpips
import os
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params

def fitNet(x, crop_orNot=False, patchSize=128):
    c, h, w = x.shape
    if crop_orNot:
        crop_obj = torchvision.transforms.CenterCrop([patchSize, patchSize])
        x = crop_obj(x)
    else:
        if (h%32 != 0) | (w%32 != 0):
            crop_obj = torchvision.transforms.CenterCrop([int(h/32)*32, int(w/32)*32])
            x = crop_obj(x)

        if (h > 800) | (w > 800):
            print("!!!!! so big !!!!!!")
            crop_obj = torchvision.transforms.CenterCrop([512, 512])
            x = crop_obj(x)

    return x


# !/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch
import cv2


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # # RGB转BRG
    # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)

def get_args():
    #学习率递减策略1: [40,50,60,70,80,90] gmma=0.5
    #学习率递减策略2: 余弦退火算法递减
    parser = argparse.ArgumentParser(description="NLAQNet_train")
    parser.add_argument("--cropPatch", type=bool, required=False, default=False, help="Crop Patchsize or Not")
    parser.add_argument("--pachSize", type=int,required=False, default=128,help="Training batch size")

    parser.add_argument("--read_weights", type=str, required=False, default="./pretrain_model/model_best_Mixed.pth", help='path of load weights files')  # "./models/Rain100H/学习率0.001/"
    parser.add_argument("--test_dataset", type=str, required=False, default='Rain100H', help='path to test data')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()
    if torch.cuda.is_available():
        save = torch.load(opt.read_weights)
    else:
        save = torch.load(opt.read_weights, map_location=torch.device('cpu'))
    model = QDerainNet().to(device)
    model.load_state_dict(save['state_dict'])

    cnn_paras_count(model)
    # input_Path = "../dataset/rain/"
    target_Path = "./images/"   #target/
    derain_Path = "./images/"   #derain/
    # testDir_List = ['Rain100H', 'Rain100L', 'test100', 'test1200', ]
    testDir_List = []
    testDir_List.append(opt.test_dataset)
    testData_PSNR = []
    testData_SSIM = []
    testData_lpips = []

    testdata_path = "./data/mixtest/"
    for dataset in testDir_List:
        print("dataset: ", dataset)
        target_Dir = target_Path + dataset + "/target/"
        derain_Dir = derain_Path + dataset + "/derain/"
        test_dir = testdata_path + dataset
        inp_fileList, tar_fileList = DataLoader_Test(testdata_path+dataset)
        SSIM_loss = SSIM()
        PSNR_TEST = 0.0
        SSIM_TEST = 0.0
        show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
        num = 0

        model.eval()
        lpips_val = []
        lpips_loss = lpips.LPIPS(net='alex')
        lpips_loss.eval()
        PSNR_list = []
        SSIM_list = []
        timeSPP = []
        start_time = time.time()
        for i in range(0, len(inp_fileList)): #len(inp_fileList)
            print("i: ", i)
            index = i
            num += 1

            inp_img = cv2.imread(inp_fileList[index])
            tar_img = cv2.imread(tar_fileList[index])
            imgname = inp_fileList[index][inp_fileList[index].find("rain")+5:]


            input_ = TF.to_tensor(inp_img)
            target = TF.to_tensor(tar_img)
            # print("shape:", input_.shape, target.shape)
            input_ = fitNet(input_, opt.cropPatch,opt.pachSize)
            target = fitNet(target, opt.cropPatch,opt.pachSize)
            # print("fitNet  :", input_.shape, target.shape)

            with torch.no_grad():

                input_ = torch.unsqueeze(input_, dim=0)
                epoch_start_time = time.time()
                input_ = input_.to(device)
                restored = model(input_)
                # restored = restored.to("cpu")
                time_spp = time.time() - epoch_start_time
                timeSPP.append(time_spp)
                target = torch.unsqueeze(target, dim=0)
                psnr = PSNR(restored, target)
                PSNR_list.append(psnr.detach().numpy())
                ssim = SSIM_loss(restored, target)
                SSIM_list.append(ssim.detach().numpy())
                lpips_val.append(lpips_loss(restored, target))

                # # 保存图像
                save_image_tensor2cv2(input_tensor=target, filename=os.path.join(target_Dir, imgname))
                save_image_tensor2cv2(input_tensor=restored, filename=os.path.join(derain_Dir, imgname))


        PSNR_TEST = np.mean(PSNR_list)
        SSIM_TEST = np.mean(SSIM_list)
        print("ave_psnr: ", PSNR_TEST)
        print("ave_ssim: ", SSIM_TEST)
        avg_lpips = sum(lpips_val)/num
        print('LPIPS: %.4f' % (avg_lpips))
        testData_PSNR.append(PSNR_TEST)
        testData_SSIM.append(SSIM_TEST)
        testData_lpips.append(avg_lpips)

        ave_timeSPP = sum(timeSPP)/num
        print("ave_timeSPP: ", ave_timeSPP)
        print("pureAll_timeSPP: ", sum(timeSPP))
        print("all_time: ", time.time() - start_time)




