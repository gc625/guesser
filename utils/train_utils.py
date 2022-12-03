from torch.utils.data.dataloader import DataLoader
from dataset import USLocations
from timm.models import create_model
import ConvNeXt.models.convnext
import ConvNeXt.models.convnext_isotropic
import torch.nn as nn
import torch
from pathlib import Path
from utils.model_utils import BinBasedConVNeXt, CLSREG_loss







def setup_model(ckpt,with_bin,pretrained_IM22k,logger):

    model = create_model(
    'convnext_base', 
    pretrained=True,
    in_22k=True, 
    num_classes=21841, 
    drop_path_rate=0,
    layer_scale_init_value=1e-6,
    head_init_scale=1,
    )


    if with_bin:
        model = BinBasedConVNeXt(base_model=model,mlp_channels=[256],classes=51)

    else:
        model.head = nn.Linear(1024,2)

    if ckpt != '' and Path(ckpt).exists():
        if logger is not None:
            logger.info(f'ckpt from {ckpt} loaded')
        print(f'ckpt from {ckpt} loaded')
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt)
    else:
        if logger is not None:
            logger.info(f'no ckpt loaded')
        print(f'no ckpt loaded')
        


    return model


def setup_loss_function(with_bin=False,mean_positions=None):

    if with_bin:
        if mean_positions == None:
            raise ValueError('mean_positions cannot be none')

        return CLSREG_loss(mean_positions)
        
    else:
        return torch.nn.SmoothL1Loss(reduction='mean')




def create_datasets_and_loaders(images_dir,labels_dir,with_bin,batch_size):

    train_set = USLocations(images_dir=images_dir,
        labels_dir=labels_dir,
        split='train',
        withbin=with_bin)

    val_set = USLocations(images_dir=images_dir,
        labels_dir=labels_dir,
        split='val',
        withbin=with_bin)

    test_set = USLocations(images_dir=images_dir,
        labels_dir=labels_dir,
        split='test',
        withbin=with_bin)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )


    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    all_datasets = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }
    
    all_data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    return all_datasets,all_data_loaders