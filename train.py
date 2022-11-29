# %%
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
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


batch_size = 7
num_epochs = 80



train_set = USLocations(images_dir='/home/gabriel/guesser/data/images',
    labels_dir='/home/gabriel/guesser/data/labels',
    split='train')

val_set = USLocations(images_dir='/home/gabriel/guesser/data/images',
    labels_dir='/home/gabriel/guesser/data/labels',
    split='val')

test_set = USLocations(images_dir='/home/gabriel/guesser/data/images',
    labels_dir='/home/gabriel/guesser/data/labels',
    split='test')

train_loader = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)


val_loader = DataLoader(
    dataset=val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

test_loader = DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8
)


from utils.train_utils import setup_model
ckpt = '/home/gabriel/guesser/ckpts/debug_new/1669484372/epoch20_ckpt.pth'
model = setup_model(ckpt,True,None).to('cuda')



# model.head = nn.Linear(1024,2)

loss_function = torch.nn.SmoothL1Loss(reduction='mean')

cls_loss = torch.nn.NLLLoss()



optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.5)

scaler = torch.cuda.amp.GradScaler()


# bin /home/gabriel/guesser/ckpts/debug_new/1669484372/epoch20_ckpt.pth
# naiive 'ckpts/epoch11'


# model.load_state_dict(ckpt)

model.to('cuda')


# ret = []

predictions = []
labels = [] 

# %%
print('??')
with torch.no_grad():
    model.eval()    
    for batch_images,batch_labels in tqdm(test_loader):
        ## TODO: 
        # with torch.cuda.amp.autocast():
        batch_images,batch_labels = batch_images.to('cuda'),batch_labels.to('cuda')
        output = model(batch_images)

        predictions.append(output)
        labels.append(batch_labels)
            # ret.append([output,batch_labels])
    print('??')
#%%
from utils.train_utils import CLSREG_loss

L = CLSREG_loss(train_set.mean_pos)


p = [L.output_decoder(output) for output in predictions]

#%%
p
#%%
predictions = torch.concat(p)
labels = torch.concat(labels)


#%%
pdist = nn.PairwiseDistance(p=2)


distances = pdist(labels,predictions)





worst_dist, worst_idx = torch.topk(distances,10)
best_dist,best_idx = torch.topk(distances,10,largest=False)

# torch.topk()
# %%
distances.mean()
# %%
import matplotlib.pyplot as plt 



plt.hist(distances.cpu(),bins=60)
# %%
best_idx[0]
# %%
test_set.get_image(worst_idx[9])
# %%
for idx in best_idx:
    print(test_set.get_image(idx)[0],distances[idx].item())
    print(f'label: {labels[idx].cpu().numpy()},pred: {predictions[idx].cpu().numpy()}')
    display(test_set.get_image(idx)[1])
# %%
for idx in worst_idx:
    print(test_set.get_image(idx)[0])
    display(test_set.get_image(idx)[1])
# %%

import numpy as np


np.save('worst_pred.npy',predictions[worst_idx].cpu().numpy())

np.save('worst_labels.npy',labels[worst_idx].cpu().numpy())
# labels[worst_idx].cpu().numpy()
# 
# %%


np.save('best_pred.npy',predictions[best_idx].cpu().numpy())

np.save('best_labels.npy',labels[best_idx].cpu().numpy())
# labels[worst_idx].cpu().numpy()
# %%
predictions[best_idx]
# %%
set(val_set.state_idx).intersection(set(test_set.state_idx))
# %%

