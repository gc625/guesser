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



model = create_model(
    'convnext_base', 
    pretrained=True,
    in_22k=True, 
    num_classes=21841, 
    drop_path_rate=0,
    layer_scale_init_value=1e-6,
    head_init_scale=1,
    )


model.head = nn.Linear(1024,2)

loss_function = torch.nn.SmoothL1Loss(reduction='mean')

cls_loss = torch.nn.NLLLoss()



optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.5)

scaler = torch.cuda.amp.GradScaler()



ckpt = torch.load('ckpts/epoch11')


model.load_state_dict(ckpt)

model.to('cuda')


# ret = []

predictions = []
labels = [] 

# %%
# loss = 
## VALIDATION ##
with torch.no_grad():
    
    for batch_images,batch_labels in tqdm(test_loader):
        ## TODO: 
        with torch.cuda.amp.autocast():
            batch_images,batch_labels = batch_images.to('cuda'),batch_labels.to('cuda')
            output = model(batch_images)

            predictions.append(output)
            labels.append(batch_labels)
            # ret.append([output,batch_labels])


predictions = torch.concat(predictions)
labels = torch.concat(labels)


#%%
pdist = nn.PairwiseDistance(p=2)


distances = pdist(labels,predictions)





worst_dist, worst_idx = torch.topk(distances,10)
best_dist,best_idx = torch.topk(distances,10,largest=False)

torch.topk()