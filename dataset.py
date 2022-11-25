import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
from pathlib import Path 
from torchvision import transforms
from timm.models import create_model
import ConvNeXt.models.convnext
import ConvNeXt.models.convnext_isotropic   


class USLocations(Dataset):

    def __init__(self,
    images_dir: str,
    labels_dir: str,
    split: str,
    withbin: bool = False,) -> None:
        super().__init__()

        self.split = split
        self.images_dir = os.path.join(images_dir,split)
        self.coordinates = self.read_file(os.path.join(labels_dir,f"{split}_coords.txt"))
        self.coordinates = [c.split(',') for c in self.coordinates]
        self.coordinates = [[float(c[0]),float(c[1])] for c in self.coordinates]
        self.state_idx = self.read_file(os.path.join(labels_dir,f"{split}_state_idx.txt"))
        self.state_idx = [''.join(state.split(',')) for state in self.state_idx]
        self.image_paths = glob.glob(os.path.join(self.images_dir,"*"))

        self.withbin = withbin

        self.stateidx_to_coordinates = {}
        for i in range(len(self.coordinates)):
            self.stateidx_to_coordinates[self.state_idx[i]] = self.coordinates[i]

        self.toTensor = transforms.ToTensor()
        self.get_average_pos()
        snames = list(self.states.keys())
        self.state2class = {snames[i]: i for i in range(len(snames))}
        # print(state2class['az'])




    def read_file(self,file):
        
        contents = []
        with open(file) as f:
            for line in f:
                contents += [line.rstrip()]

        return contents


    def get_image(self,idx):
        image = Image.open(self.image_paths[idx])
        name = Path(self.image_paths[idx]).stem
        return name,image




    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx])        
        name = Path(self.image_paths[idx]).stem
        image_tensor = self.toTensor(image)

        coordinates = self.stateidx_to_coordinates[name]
        abs_coordinates = torch.tensor((coordinates))

        cur_state = name[:2]

        image_class = torch.tensor(self.state2class[cur_state])

        relative_coordinates = abs_coordinates - self.mean_pos[cur_state]

        if self.withbin:
            return image_tensor,image_class,relative_coordinates


        else:
            return image_tensor,abs_coordinates

    def __len__(self):

        return len(self.image_paths)



    def get_average_pos(self):
        from collections import defaultdict

        self.states = defaultdict(list)
        
        for state in self.state_idx:
            self.states[state[:2]] += [self.stateidx_to_coordinates[state]] 

        self.mean_pos = {}

        for k,v in self.states.items():
            self.mean_pos[k] = torch.tensor(v).mean(dim=0)
        
        
        




if __name__ == "__main__":
    train_dataset = USLocations(images_dir='/home/gabriel/guesser/data/images',
    labels_dir='/home/gabriel/guesser/data/labels',
    split='train')

    train_dataset.get_average_pos()
    train_dataset[2]




    # model = create_model(
    # 'convnext_large', 
    # pretrained=True,
    # in_22k=True, 
    # num_classes=21841, 
    # drop_path_rate=0,
    # layer_scale_init_value=1e-6,
    # head_init_scale=1,
    # 
    # from tqdm import tqdm
    # for i in tqdm(range(len(train_dataset))):
    #     if train_dataset[i] is not None:
    #         errors += [train_dataset[i]]

    print('')
