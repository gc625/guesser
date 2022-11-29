from torch.utils.data.dataloader import DataLoader
from dataset import USLocations
from timm.models import create_model
import ConvNeXt.models.convnext
import ConvNeXt.models.convnext_isotropic
import torch.nn as nn
import torch
from pathlib import Path



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x




class CLSREG_loss(nn.Module):
    def __init__(self,mean_positions):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = torch.nn.SmoothL1Loss(reduction='mean') 
        self.smax = nn.Softmax(dim=-1)
        self.mean_positions = []
        
        for m in mean_positions.keys():
            self.mean_positions += [mean_positions[m]]
        self.mean_positions = torch.stack(self.mean_positions).to('cuda')

    
    def get_abs_pos(self,predicted_class,predicted_offsets):


        predicted_mean_size = self.mean_positions[predicted_class]

        predicted_abs_position = predicted_mean_size+predicted_offsets

        # test = torch.split(output,2)

        return predicted_abs_position

    def output_decoder(self,output):
        smax = nn.Softmax(dim=-1)
        cls_pred,reg_pred = output['cls_pred'],output['reg_pred']
        cls_probs = smax(cls_pred)
        _,prediced_classes = cls_probs.max(dim=-1)
        B,C = reg_pred.shape
        batch_idx = torch.arange(B)
        selected_offsets = reg_pred.reshape(B,C//2,2)[batch_idx,prediced_classes]
        predicted_abs_pos = self.get_abs_pos(prediced_classes,selected_offsets)
        
        return predicted_abs_pos

    def forward(self,output,labels,writer,cur_iters):
        '''
        output['cls_pred']: (N,51)
        output['reg_pred]: (N,51*2)

        labels: (N,3): where (N,0) is true class
                             (n,1:3) is true absolute position
        '''
        
        cls_pred = output['cls_pred']
        predicted_abs_pos = self.output_decoder(output)


        true_class, true_abs_pos = labels[:,0].long(),labels[:,1:3]


        cls_loss = self.cls_loss(cls_pred,true_class)
        reg_loss = self.reg_loss(predicted_abs_pos,true_abs_pos)

        writer.add_scalar("Loss/cls_iter", cls_loss, cur_iters)
        writer.add_scalar("Loss/reg_iter", reg_loss, cur_iters)

        return cls_loss+reg_loss
        # return cls_loss




class BinBasedConVNeXt(nn.Module):
    def __init__(self,base_model,
    mlp_channels=[256],classes=51):
        super().__init__()

        self.base_model = base_model
        self.base_model.head = Identity()


        # self.cls_branch = nn.Sequential(
        #     nn.Linear(1024,classes),

        # )
        self.reg_branch = nn.Sequential(
            nn.Linear(1024,mlp_channels[0]),
            nn.BatchNorm1d(mlp_channels[0]),
            nn.ReLU(),
            nn.Linear(mlp_channels[0],2*classes)
        )

        self.cls_branch = nn.Sequential(
            nn.Linear(1024,mlp_channels[0]),
            nn.BatchNorm1d(mlp_channels[0]),
            nn.ReLU(),
            nn.Linear(mlp_channels[0],classes)
        )

    def forward(self,input):

        base_features = self.base_model(input)

        cls_preds = self.cls_branch(base_features)
        reg_preds = self.reg_branch(base_features)

        output = {
            'cls_pred': cls_preds,
            'reg_pred': reg_preds
        }
        return output




def setup_model(ckpt,with_bin,logger):

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


    return model






def setup_loss_function(with_bin=False,mean_positions=None):

    if with_bin:
        if mean_positions == None:
            raise ValueError('mean_positions cannot be none')

        return CLSREG_loss(mean_positions)
        # return torch.nn.CrossEntropyLoss()

        
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