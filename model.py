import torch
from torchvision import datasets, transforms, models
import torchvision
from timm.models import create_model
import ConvNeXt.models.convnext
import ConvNeXt.models.convnext_isotropic
from PIL import Image
from utils.convert_imagenet_label import IN22k_labels
from dataset import USLocations
import torch.optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


batch_size = 7
num_epochs = 80
with_bin = True





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
    shuffle=True,
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

model.to('cuda')



loss_epochs = []
val_epochs = []
for epoch in range(num_epochs):
    train_bar = tqdm(train_loader)
    total_loss = 0


    ## TRAINING ## 
    for i,data in enumerate(train_bar):
        batch_images,batch_labels = data
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            batch_images,batch_labels = batch_images.to('cuda'),batch_labels.to('cuda')
            output = model(batch_images)

            # calc loss
            loss = loss_function(output,batch_labels)
            writer.add_scalar("Loss/train_iter", loss, (i*batch_size)+(epoch*len(train_bar)))
        # propagate loss
        scaler.scale(loss).backward()

        # update the parameters
        scaler.step(optimizer)
        total_loss+= loss.item()
        train_bar.set_postfix({'loss':f'{loss.item():.2f}','epoch':epoch})
    

        scaler.update()
        writer.add_scalar("lr",scheduler.get_last_lr()[0],(epoch+1)*i)
    
    
    scheduler.step()
    ## logging training statistics
    average_loss = total_loss/len(train_loader)
    loss_epochs += [average_loss]
    writer.add_scalar("Loss/train_epoch", average_loss, (epoch+1))
    print(f'Epoch {epoch} average loss: {average_loss:.2f}')
    
    ## VALIDATION ##
    with torch.no_grad():
        total_loss = 0 
        for batch_images,batch_labels in tqdm(val_loader):
            ## TODO: 
            with torch.cuda.amp.autocast():
                batch_images,batch_labels = batch_images.to('cuda'),batch_labels.to('cuda')
                output = model(batch_images)
                loss = loss_function(output,batch_labels)
            total_loss += loss


        ## logging validation statistics
        val_loss =  total_loss/len(val_loader.dataset)
        val_epochs += [val_loss]
        print(f'Epoch {epoch}, val acc; {val_loss:2f}')
        writer.add_scalar("Loss/val_epoch", val_loss, (epoch+1))

    torch.save(model.state_dict(), f'/home/gabriel/guesser/ckpts/epoch{epoch}')

    