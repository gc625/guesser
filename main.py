import torch
from torchvision import datasets, transforms, models
import torchvision
from timm.models import create_model
import ConvNeXt.models.convnext
import ConvNeXt.models.convnext_isotropic
from PIL import Image
from convert_imagenet_label import IN22k_labels
from dataset import USLocations
import torch.optim
import torch.nn as nn
from utils.train_utils import create_datasets_and_loaders,setup_model,setup_loss_function
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from pathlib import Path
import logging
import sys
import time
import calendar

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def setup_custom_logger(name,write_dir):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(os.path.join(write_dir,'log.txt'), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=7, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--scheduler', default='linear', type=str, metavar='scheduler',
                        help='scheduler (default: "linear"')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--max_norm', type=float, default=15, metavar='norm',
                        help='max grad norm')
    
    parser.add_argument('--ckpt', type=str, default='ckpts/debug_new/1669476920/epoch1_ckpt.pth', metavar='ckpt',
                        help='ckpt path')
    
    parser.add_argument('--with_bin', type=str2bool, default=True,
                        help='use model with bin+regression')
    
    parser.add_argument('--mixed_precision', type=str2bool, default=True,
                        help='train model with mixed precision')
    
    parser.add_argument('--eval_freq',default=1,type=int,help='how often to eval during training')
    
    parser.add_argument('--output_dir',type=str,default='ckpts',help='where to write ckpts')
    
    parser.add_argument('--extra_tag',type=str,default='debug_new',help='subfolder to put ckpts and logs')

    return parser


def main(args):
    print(args)


    #@ ARGS
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    ckpt = args.ckpt
    with_bin = args.with_bin
    mixed_precision = args.mixed_precision
    eval_freq = args.eval_freq
    out_dir = args.output_dir
    max_norm = args.max_norm
    opt = args.opt
    sched = args.scheduler
    extra_tag = args.extra_tag
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)


    cur_dir = os.path.join(out_dir,extra_tag,str(time_stamp))
    os.makedirs(cur_dir,exist_ok=True)

    logger = setup_custom_logger('trainlogger',cur_dir)
    logger.info(args)
    #@ Assigning stuff
    datasets, dataloaders = create_datasets_and_loaders(
        '/home/gabriel/guesser/data/images',
        '/home/gabriel/guesser/data/labels',
        batch_size=batch_size,
        with_bin=with_bin
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    model = setup_model(ckpt,with_bin,logger).to('cuda')

    # if with_bin:
    loss_function = setup_loss_function(with_bin,mean_positions=train_loader.dataset.mean_pos)
    if with_bin:
        output_decoder = loss_function.output_decoder
        # output_decoder = None
    else:
        output_decoder = None
    
    if opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    
    if sched == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,1,0.1,total_iters=20)
    elif sched == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.5)
    pdist = nn.PairwiseDistance(p=2)

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()

    logger.info(f'train set length: {len(train_loader)}')
    logger.info(f'val set length: {len(val_loader)}')
    logger.info(f'test set length: {len(test_loader)}')
    
    logger.info(f'MODEL: {model}')
    logger.info(f'LOSS FUNCTION: {loss_function}')
    logger.info(f'optimizer: {optimizer}')
    logger.info(f'scheduler: {scheduler}')

    writer.add_scalar("LR/lr",scheduler.get_last_lr()[0],0)

    

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_bar = tqdm(train_loader)
        for iter,data in enumerate(train_bar): 

            optimizer.zero_grad()
            images,labels = data[0].to('cuda'),data[1].to('cuda')
            cur_iters = (iter*batch_size)+(epoch*len(train_bar))

            output = forward_step(
                images,
                model,
                mixed_precision)

            loss = backward_step(
                model,
                output,
                labels,
                loss_function,
                optimizer,
                mixed_precision,
                with_bin,
                scaler,
                writer,
                cur_iters,
                max_norm
            )

            epoch_loss += loss.item()

            writer.add_scalar("LR/lr",scheduler.get_last_lr()[0],cur_iters)
            writer.add_scalar("Loss/train_iter", loss, cur_iters)
            train_bar.set_postfix({'loss':f'{loss.item():.2f}','epoch':epoch})
        
        scheduler.step()        
        average_loss = epoch_loss/len(train_loader)

        writer.add_scalar("Loss/train_epoch", average_loss, epoch)
        # logger.info()

        torch.save(model.state_dict(),os.path.join(cur_dir,f'epoch{epoch}_ckpt.pth'))
        logger.info(f"ckpt saves to {os.path.join(cur_dir,f'epoch{epoch}_ckpt.pth')}")
        if epoch % eval_freq == 0:
            model.eval()
            total_loss = 0 
            with torch.no_grad():
                for images,labels in tqdm(val_loader):
                    images,labels = images.to('cuda'),labels.to('cuda')    

                    loss = val_step(
                        images=images,
                        labels=labels,
                        model=model,
                        val_metric=pdist,
                        mixed_precision=mixed_precision,
                        output_decoder=output_decoder
                    )

                    total_loss += loss.sum().item()
                
                val_loss =  total_loss/len(val_loader.dataset)
                writer.add_scalar("Loss/val_epoch", val_loss, (epoch))

        

def forward_step(images,model,mixed_precision):
    if mixed_precision:
        with torch.cuda.amp.autocast():
            output = model(images)

    else:
        output = model(images)
        
    return output
        

def val_step(images,labels,model,val_metric,mixed_precision,output_decoder):
    
    if mixed_precision:
        with torch.cuda.amp.autocast():
            output = model(images)
            
            if output_decoder is not None:
                output = output_decoder(output)
                loss = val_metric(output,labels[:,1:3])
            else:
                loss = val_metric(output,labels)

    else:
        output = model(images)

        if output_decoder is not None:
            output = output_decoder(output)
            loss = val_metric(output,labels[:,1:3])
        else:
            loss = val_metric(output,labels)


    return loss 


def backward_step(model,output,labels,loss_function,optimizer,mixed_precision,with_bin,scaler,writer,cur_iters,max_norm):

    if mixed_precision:
        with torch.cuda.amp.autocast():
            if with_bin:
                loss = loss_function(output,labels,writer,cur_iters)
                # loss = loss_function(output,labels)
            else:
                loss = loss_function(output,labels)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
    else:
        if with_bin:
            # loss = loss_function(output,labels)
            loss = loss_function(output,labels,writer,cur_iters)
        else:
            loss = loss_function(output,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm)
        optimizer.step()
    return loss
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)





