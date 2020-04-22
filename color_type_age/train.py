# Author: Jia Xu
# modified from https://github.com/jiweibo/ImageNet

import argparse
import os
import time
import shutil
import torch

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import torch.utils.data as data

import torchvision.models as models
#from data_set import get_dataset
from utils import Bar, Logger, AverageMeter, accuracy
from my_dataset import *

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ChineseFood Training')
parser.add_argument('--task', type=str,default='skin_type', help='Tasks')
parser.add_argument('--path', default='/data/yd_data/skin-quality/bounded_skin_data/imgs', help='path to dataset')
parser.add_argument('--txt_path', type=str,default='/data/xujia/skin_quality/new_labels/', help='path to label txt,ori:processed_labels')
parser.add_argument('--label_interval',type=int,default=1)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')

parser.add_argument('--input-size', default=224, type=int, metavar='N',
                    help='image size for the input of DNN (default: 224)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to load latest checkpoint, (default: None)')
parser.add_argument('--save-dir', default='./new_data/', type=str,
                    help='path to save checkpoints , (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#parser.add_argument('--nlabel', default=90000, type=int, help='number of labels')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

best_prec1 = 0.0
args = parser.parse_args()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
#use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    global args, best_prec1

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    # create model
    # change the fully-connected layer for Food (208 classes)
    model = getattr(models, args.arch)(pretrained=args.pretrained)
    # for param in model.parameters():
    #     param.requires_grad_(requires_grad=False)
    num_ftrs = model.fc.in_features
    if args.task == 'skin_type':
        num_classes = 4
    elif args.task == 'skin_color':
        num_classes = 6
    elif args.task == 'skin_age':
        num_classes = (int(67//args.label_interval)+1)
        print("num_classes",num_classes)
    model.fc = nn.Linear(num_ftrs, num_classes)

    # use cuda
    model = nn.DataParallel(model).to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    # optionlly resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        args.save_dir = os.path.join(args.save_dir, args.arch)
        args.save_dir = os.path.join(args.save_dir, args.task)
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=args.task, resume=True)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        args.save_dir = os.path.join(args.save_dir, args.arch)
        args.save_dir = os.path.join(args.save_dir,args.task)
        if not os.path.exists(args.save_dir):os.makedirs(args.save_dir)
        print("args.save_dir ",args.save_dir )
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=args.task)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Data loading
    augmentation = dataAugmentation()
    txt_name = 'train_' + args.task +'.txt' #eg. train_skin_color.txt
    train_txt = os.path.join(args.txt_path,txt_name)
    trainset = dataLoader(train_txt, args.path,dataset="trainImages", label_interval = args.label_interval,data_transforms=augmentation.data_transforms)

    txt_name = 'val_' + args.task + '.txt'  # eg. train_skin_color.txt
    val_txt = os.path.join(args.txt_path, txt_name)
    valset = dataLoader(val_txt, args.path, dataset="testImages", label_interval = args.label_interval,data_transforms=augmentation.data_transforms)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True,
                                   num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch,
                                 shuffle=False,
                                 num_workers=args.workers)



    if args.evaluate:
        txt_name = 'test_' + args.task + '.txt'
        val_txt = os.path.join(args.txt_path, txt_name)
        valset = dataLoader(val_txt, args.path, dataset="testImages",label_interval=args.label_interval, data_transforms=augmentation.data_transforms)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch,
                                 shuffle=False,
                                 num_workers=args.workers)

        validate(val_loader, model, criterion)
        return



    hyper_parameter_str = '{}_{}'.format(args.arch, str(args.label_interval))
    sub_save_dir = os.path.join(args.save_dir, hyper_parameter_str)
    print("sub_save_dir",sub_save_dir)
    if not os.path.exists(sub_save_dir): os.makedirs(sub_save_dir)
        
        
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion,epoch)

        scheduler.step()

        lr = 0
        logger.append([lr, train_loss, val_loss, train_acc, val_acc])

        # remember the best prec@1 and save checkpoint
        is_best = val_acc > best_prec1
        best_prec1 = max(val_acc, best_prec1)

        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, checkpoint=sub_save_dir)

    logger.close()


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (input, target) in enumerate(train_loader):
        #print(" batch_idx, (input, target)", batch_idx, input.shape, len(target))
        #print(target)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = 'Epoch: ({epoch}/{total1})|({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}| top3:{top3:.2f}'.format(
            epoch=epoch+1,
            total1=args.epochs,
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top3=top3.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()

    # switch to evaluate mode
    model.eval()

    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = 'Epoch: ({epoch}/{total1})|({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}| top3: {top3:.2f}'.format(
                epoch=epoch+1,
                total1=args.epochs,
                batch=batch_idx + 1,
                size=len(val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top3=top3.avg)
            bar.next()
    bar.finish()
    return losses.avg, top1.avg

    

if __name__ == '__main__':
    main()