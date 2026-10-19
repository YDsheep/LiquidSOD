import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from models.LiquidSOD import LiquidSOD
from data import get_loadere, test_dataset
from thop import profile
import torch.nn as nn
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from tqdm import tqdm
from collections import OrderedDict
from utils.init_func import group_weight,clip_gradient, adjust_lr
import random
import IOU
from evaluation import metric as M
IOU = IOU.IOU(size_average=True)

# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True
p = OrderedDict()
p['lr'] = 1e-4  # Learning rate
p['wd'] = 0.01  # Weight decay
p['momentum'] = 0.90  # Momentum

# set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loadere(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("MambaSOD-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()

step = 0
writer = SummaryWriter(save_path + 'summary')
best_SM = 0
best_epoch = 0


def load_backbone_weights(model, weight_path):
    print(f"Loading backbone weights from {weight_path}...")
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        model_dict = model.state_dict()


        pretrained_dict = {}
        matched_count = 0

        for k, v in state_dict.items():
            for prefix in ['', 'backbone.', 'encoder.', 'context_encoder.']:
                target_key = prefix + k
                if target_key in model_dict:
                    if model_dict[target_key].shape == v.shape:
                        pretrained_dict[target_key] = v
                        matched_count += 1
                        break

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Successfully loaded {matched_count} backbone layers.")

    except FileNotFoundError:
        print(f"Error: Pretrained weights not found at {weight_path}")

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, )
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def structure_loss2(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask)
    return bce+iou


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        loop = tqdm(enumerate(train_loader, start=1), total=len(train_loader), ncols=100, desc=f"Epoch {epoch:03d}",
                    leave=False)

        for i, (images, gts) in loop:
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            s, s1, s2 = model(images)

            loss0 = structure_loss(s, gts)
            loss1 = structure_loss(s1, gts)
            loss2 = structure_loss(s2, gts)
            # loss3 = structure_loss(s3, gts)

            loss = loss0 + loss1 + loss2

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data

            loop.set_postfix(loss=float(loss.data), sal=float(loss0.data))
            if i % 100 == 0 or i == len(train_loader) or i == 1:
                loop.write('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} salLoss: {:0.4f}'.format(
                    epoch, opt.epoch, i, len(train_loader), loss.data, loss0.data))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), save_path + 'MambaSOD_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'MambaSOD_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def val(test_loader, model, epoch, save_path):
    global best_SM, best_epoch
    Sm_fun = M.Smeasure()
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            s, s1, s2 = model(image)
            res = F.interpolate(s, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            Sm_fun.step(pred=res, gt=gt)
        sm = Sm_fun.get_results()['sm']
        writer.add_scalar('SM', torch.tensor(sm), global_step=epoch)
        print('Epoch: {} SM: {} ####  bestSM: {} bestEpoch: {}'.format(epoch, sm, best_SM, best_epoch))
        if epoch == 1:
            best_SM = sm
        else:
            if epoch == 1 or sm > best_SM:
                best_SM = sm
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'MambaSOD_epoch_best.pth')
                print('best_train epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} SM:{} bestEpoch:{} bestSM:{}'.format(epoch, sm, best_epoch, best_SM))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# build the model
model = WLNet()

if opt.load is not None:
    model.load_state_dict(torch.load(opt.load))
    print('Resuming training from checkpoint:', opt.load)
elif opt.pre is not None:
    load_backbone_weights(model, opt.pre)

model = model.cuda()

if __name__ == '__main__':
    print("Start train...")
    set_seed(1024)
    model = model.cuda()
    params = model.parameters()
    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, p['lr'])
    optimizer = torch.optim.AdamW(params_list, lr=p['lr'], betas=(0.9, 0.999), weight_decay=p['wd'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-5,
        last_epoch=-1
    )


    input1 = torch.randn(1, 3, 448, 448)
    input1  = input1.cuda()
    flops, params = profile(model, inputs=(input1,))
    print(f"FLOPs: {flops}, Params: {params}")

    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch, save_path)
        val(test_loader, model, epoch, save_path)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
