import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.a.a1_8 import MINet_Res50

# from utils.loss import SegmentationLosses
# from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
# from utils.CEL import CEL
import utils.ssim_loss as ssim_loss
import utils.iou_loss as iou_loss
# import utils.fb as fb_loss

import torch

class Trainer(object):
    def __init__(self, args):
        self.args = args


        # self.writer = self.summary.create_summary()
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        model = MINet_Res50()
        
        train_params = [{'params': model.get_1x_lr_params(), 'lr': 1e-5},
                        {'params': model.get_10x_lr_params(), 'lr': 1e-5}]

        # Define Optimizer
        # optimizer = torch.optim.Adam(train_params, lr = args.lr, weight_decay=5.e-4)
        optimizer = torch.optim.AdamW(train_params, lr = args.lr, weight_decay=5.e-2)  # 
        


        self.model,  self.optimizer = model, optimizer

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        self.model.module.load_state_dict(torch.load('/home/siyujiao/ex2/v3/run/duts/a1_8/again/model_best.pth.tar')['state_dict'])

    def validation(self, epoch):
        self.model.eval()

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        mae = 0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            tbar.set_description('Test loss: %.3f' % (i))

            pred = output.squeeze(1).cpu().numpy()
            target = target.cpu().numpy()

            mae = mae + np.mean(np.abs(pred-target))
        mae = mae/i

        new_pred = mae
        print("-----------------------mae:", mae, " -----------------------")



def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=12,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='duts',
                        choices=['pascal', 'duts', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=5,                  ###
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=384,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=384,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=20,               ###########
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--loss', type=str, default='ssim',
                        help='choose loss u need, from [ssim,CLE] ')
    parser.add_argument('--use_aux', type=bool, default=False,
                        help='whether to use aux loss')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,  # 5e-4
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',       
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='rgb/a1_rgb8/1',                  #####
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')




    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    trainer.validation(1)



if __name__ == "__main__":
   main()
