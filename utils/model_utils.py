from torch.utils.data.dataloader import DataLoader
from dataset import USLocations
from timm.models import create_model
import ConvNeXt.models.convnext
import ConvNeXt.models.convnext_isotropic
import torch.nn as nn
import torch
from pathlib import Path




class CLSREG_loss(nn.Module):
    '''
    Custom loss function that combines a CE Loss and regression loss

    '''
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
        return predicted_abs_position

    def output_decoder(self,output,return_class_pred=False):
        smax = nn.Softmax(dim=-1)
        cls_pred,reg_pred = output['cls_pred'],output['reg_pred']
        cls_probs = smax(cls_pred)
        _,prediced_classes = cls_probs.max(dim=-1)
        B,C = reg_pred.shape
        batch_idx = torch.arange(B)
        selected_offsets = reg_pred.reshape(B,C//2,2)[batch_idx,prediced_classes]
        predicted_abs_pos = self.get_abs_pos(prediced_classes,selected_offsets)
        
        if return_class_pred:
            return predicted_abs_pos,prediced_classes

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





class BinBasedConVNeXt(nn.Module):
    def __init__(self,base_model,
    mlp_channels=[256],classes=51):
        super().__init__()

        self.base_model = base_model
        self.base_model.head = nn.Identity() 


        #TODO: this is a very bad way of doing it but I have no time
        #TODO: you can loop thru the input mlp_channels and create 
        #      layers that way 

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
        '''
        combines backbone with prediction heads
        '''
        
        base_features = self.base_model(input)

        cls_preds = self.cls_branch(base_features)
        reg_preds = self.reg_branch(base_features)

        output = {
            'cls_pred': cls_preds,
            'reg_pred': reg_preds
        }
        return output
