from model.networks import QDerainNet
from model.CR import ContrastLoss
import numpy as np
import time
import datetime
import torch
from torch import nn
import torchvision
from torchvision import transforms
import os
from tqdm import tqdm
import torch.optim as optim
import utils
from utils.PSNR import torchPSNR as PSNR
from utils.SSIM import SSIM
from torch.utils.tensorboard import SummaryWriter
from utils.data_RGB import get_training_data, get_validation_data
import argparse
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params

def get_args():
    #学习率递减策略1: [40,50,60,70,80,90] gmma=0.5
    #学习率递减策略2: 余弦退火算法递减
    parser = argparse.ArgumentParser(description="NLAQNet_train")
    parser.add_argument("--batchSize", type=int,required=False, default=7,help="Training batch size")
    parser.add_argument("--pachSize", type=int,required=False, default=128,help="Training batch size")
    parser.add_argument("--epochs", type=int, required=False, default=300, help="Number of training epochs")

    parser.add_argument("--loadWeights", type=bool, default=False, help="Train use pretrained model or not")
    parser.add_argument("--read_weights", type=str, required=False, default="./pretrain_model/model_best_100H.pth", help='path of load weights files')  # "./models/Rain100H/学习率0.001/"
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--save_weights", type=str, required=False,default="./pretrain_model/", help='path of log files') #"./models/Rain100H/学习率0.001/"
    parser.add_argument("--train_data", type=str, required=False, default='./data/Rain100H/Train/', help='path to training data')
    parser.add_argument("--val_data", type=str, required=False, default='./data/mixtest/Rain100H/', help='path to valid data')
    return parser.parse_args()

if __name__ == '__main__':
    opt = get_args()
    '''############################################## load model ########################################################################'''

    model = QDerainNet().to(device)
    if opt.loadWeights:
        print("loading pretained model !!!")
        if torch.cuda.is_available():
            save = torch.load(opt.read_weights)
        else:
            save = torch.load(opt.read_weights, map_location=torch.device('cpu'))
        model.load_state_dict(save['state_dict'])

    cnn_paras_count(model)
    lr = opt.lr
    best_psnr = 0
    best_epoch = 0
    model_dir = opt.save_weights
    w_loss_l1 = 1
    w_loss_vgg7 = 0.5

    PSNRMAX = 0
    best_psnr = False
    '''############################################## load data ########################################################################'''
    NUM_EPOCHS = opt.epochs  # 训练周期
    BATCH_SIZE = opt.batchSize  # 每批次训练数量
    patch_size = opt.pachSize  # 训练图像裁剪大小
    train_dir = opt.train_data  # 训练数据集目录
    val_dir = opt.val_data

    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size': patch_size})
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=0, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size': 256})
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=0,
                                             drop_last=False, pin_memory=True)

    '''############################################## optimizer #########################################################################'''
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-6)

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)

    # loss
    criterion = []
    criterion.append(nn.MSELoss().to(device))
    criterion.append(ContrastLoss(ablation=False))
    criterion.append(SSIM().to(device))

    optimizer.zero_grad()



    writer = SummaryWriter('runs/experiment_1')
    # images = torch.rand([1, 3, 128, 128]).to(device)
    # writer.add_graph(model, images)
    # writer.close()

    for epoch in range(1, NUM_EPOCHS + 1):
        PSNR_list = []
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            for param in model.parameters():
                param.grad = None

            target = data[0].to(device)
            input_ = data[1].to(device)

            restored = model(input_)

            loss_rec = criterion[0](restored, target)
            loss_vgg7 = criterion[1](restored, target, input_)
            loss_ssim = 1 - criterion[2](restored, target)
            loss = w_loss_l1 * loss_rec + w_loss_vgg7 * loss_vgg7 + loss_ssim
            loss.backward()
            if (loss>3.0)&(epoch>30):
                for param in model.parameters():
                    param.grad = None

            train_psnr = PSNR(restored, target)
            PSNR_list.append(train_psnr.detach().cpu().numpy())

            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        with torch.no_grad():
            train_psnr = np.mean(PSNR_list)
            writer.add_scalar('loss',
                              loss,
                              epoch * len(train_loader))
            writer.add_scalar('train PSNR: ',
                              train_psnr,
                              epoch * len(train_loader))
            writer.add_scalar('lr: ',
                              optimizer.state_dict()['param_groups'][0]['lr'],
                              epoch * len(train_loader))
        scheduler.step()
        #### Evaluation ####
        if epoch % 1 == 0:
            model.eval()
            PSNR_list = []
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)

                with torch.no_grad():
                    restored = model(input_)
                psnr = PSNR(restored, target)
                PSNR_list.append(psnr.detach().cpu().numpy())
            psnr = np.mean(PSNR_list)
            writer.add_scalar('test PSNR: ',
                              psnr,
                              epoch)
            if PSNRMAX < psnr:
                PSNRMAX = psnr
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            }, os.path.join(model_dir, "model_best.pth"))


        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\ttrain_psnr: {:.4f}\ttest_psnr: {:.4f}\tLearningRate {:.8f}".format(epoch,
                                                                                                time.time() - epoch_start_time,
                                                                                                epoch_loss, train_psnr, psnr,
                                                                                                scheduler.get_lr()[0]))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))







