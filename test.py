# %%
import torch
import torchvision
from timm.models import create_model
import ConvNeXt.models.convnext
import ConvNeXt.models.convnext_isotropic
from PIL import Image
from utils.convert_imagenet_label import IN22k_labels
from dataset import USLocations
import torch.optim
import torch.nn as nn
from utils.train_utils import create_datasets_and_loaders,setup_model,setup_loss_function
from tqdm import tqdm
import numpy as np

#%%
def get_results(with_bin=False,ckpt='/home/gabriel/guesser/public_checkpoints/plain_reg.pth'):    
    batch_size = 16
    all_datasets, all_dataloaders =  create_datasets_and_loaders(
            '/home/gabriel/guesser/data/images',
            '/home/gabriel/guesser/data/labels',
            batch_size=batch_size,
            with_bin=with_bin
    )
    
    train_dataset = all_datasets['train']
    test_dataloader = all_dataloaders['test']

    model = setup_model(ckpt=ckpt,
                with_bin=with_bin,
                logger=None).to('cuda')
    
    loss_function = setup_loss_function(with_bin=with_bin,mean_positions=train_dataset.mean_pos)
    output_decoder = loss_function.output_decoder if with_bin else None

    predictions = []
    pred_class = []
    labels = []
    with torch.no_grad():
        model.eval()    
        for batch_images,batch_labels in tqdm(test_dataloader):
            batch_images,batch_labels = batch_images.to('cuda'),batch_labels.to('cuda')
            output = model(batch_images)
            if output_decoder is not None:
                predicted_abs_pos,prediced_classes = output_decoder(output,return_class_pred=True)
                predictions.append(predicted_abs_pos)
                pred_class.append(prediced_classes)
                labels.append(batch_labels)
            else:       
                predictions.append(output)
                labels.append(batch_labels)
    if with_bin:

        return torch.concat(pred_class),torch.concat(predictions),torch.concat(labels)
    else:
        return torch.concat(predictions),torch.concat(labels)
# %%
# p,l = get_results()
# # %%
# predicted = p.cpu().numpy()
# labels = l.cpu().numpy()


# np.save('results/plainreg_predictions.npy',predicted)
# np.save('results/plainreg_labels.npy',labels)

# %%
p_class,p_value,l = get_results(with_bin=True,ckpt='/home/gabriel/guesser/public_checkpoints/binbased.pth')

#%%
predicted_class = p_class.cpu().numpy()
predicted_pos = p_value.cpu().numpy()
labels = l.cpu().numpy()


np.save('results/binbased_pos.npy',predicted_pos)
np.save('results/binbased_class.npy',predicted_class)
np.save('results/binbased_labels.npy',labels)

# %%
