import requests
import json 
import glob 
from tqdm import tqdm
import os
import urllib.error
import urllib.parse
import urllib.request
import urllib
from pathlib import Path as P
import random
from itertools import chain
import numpy as np
from PIL import Image
import hashlib
import hmac
import base64
import urllib.parse as urlparse









def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()



class State():
    '''
    State class for every state in America, stores sampled indices, and coordinates    
    
    '''

    def __init__(self,file_path,state_name):
        
        self._file_path = file_path
        print(f'Creating State {state_name}')
        
        # set up @properties and aesthetic stuff
        with open(self._file_path) as f:
            lines =f.readlines()
        self._len = len(lines)
        self._name = state_name
        self.sampled_idx_file = P(self._file_path).parents[1] / 'sampled_coordinates' / f'{self._name}.txt'
        
        # Create a set to add all sampled coordinates
        self.sampled_indices = set()
        if self.sampled_idx_file.exists():
            with open(self.sampled_idx_file) as f:
                for line in f:
                    self.sampled_indices.add(int(line.rstrip()))
        
    def __str__(self) -> str:        
        return str(self._name)

    @property
    def path(self):
        return self._file_path

    def __len__(self):
        return self._len

    def __repr__(self):
        return f'State object for state {self._name}'

    def load_coordinates(self):
        
        self.coords = []
        with open(self._file_path) as f:
            self.coords += [line.strip() for line in f]
        return self.coords

    def update_sampled_indices(self,new_indices):
        self.sampled_indices = self.sampled_indices.update(new_indices)

        with open(self.sampled_idx_file,'a') as f:
            for idx in new_indices:
                f.write(f'{idx}\n')
            print(f'{len(new_indices)} new indices added to {self._name}')
        f.close()

        return self.sampled_indices




class Datatools():
    '''
    Set of tools to extract valid USA street coordinates from https://batch.openaddresses.io/data
    
    Data folder should either be in the root dir, or softlinked to the curent directory
    
    data
    ├── coordinates_by_state # 51 text files with UNPROCESSED coordinates
    ├── images # Where script extracts images
    │   ├── raw
    │   ├── test
    │   ├── train
    │   └── val
    ├── labels # 51 .txt containing SAMPLED indices in train/val/test of sampled coordinates 
    ├── raw_jsons 
    │   └── us # CONTAINS 51 folders with all the downloaded data
    └── sampled_coordinates # 51 .txt containing sampled coordinates
    '''

    def __init__(self,data_root,raw_json_dirs,coords_dir) -> None:
        self.data_root = data_root
        self.raw_json_dir = os.path.join(data_root,raw_json_dirs)
        self.coords_dir = os.path.join(data_root,coords_dir)
        self.states = sorted(glob.glob(os.path.join(self.raw_json_dir,"*/")))
        self.label_dir = os.path.join(self.data_root,"labels")
        self.bad_indices_dir = os.path.join(self.data_root,"labels",'bad_indices.txt')
        self.coordinates_with_errors_dir = os.path.join(self.data_root,'images/badcoordinates.txt')
        self.API_KEY = None
        self.SECRET = None
        self.xmin,self.xmax = 3.688565,71.242223
        self.ymin,self.ymax = -171.208898, -48.664879
        print('Loading State Data')

        # Automatically extract from raw jsons if not exist
        if len(glob.glob(os.path.join(self.coords_dir,'*'))) == 0:
            self.extract_coordinates()

        # load coordinates into a dict
        self.state_dict = {
            str(P(state).stem): State(os.path.join(self.coords_dir,P(state).stem)+'.txt',P(state).stem) for state in self.states
        }
        print('Done loading State Data')



        # ! TODO: number of coordinates way to much anyways, so set max length to be the minimum number of coordinates for any state
        self.max_dataset_size = min([len(state) for _,state in self.state_dict.items()])*len(self.state_dict)
        

        # Some coordinates are formatted incorrectly, prepare set to track these
        self.bad_indices = set()
        if P(self.bad_indices_dir).exists():
            with open(self.bad_indices_dir) as f:
                for line in f:
                    self.bad_indices.add(int(line.rstrip()))
        
        # Some coordinates return error when querying street view api, also keep track of them
        self.coordinates_with_errors = []
        if P(self.coordinates_with_errors_dir).exists():
            with open(self.coordinates_with_errors_dir) as f:
                for line in f:
                    self.coordinates_with_errors.append(str(line.rstrip()))


    
    def extract_coordinates(self):
        '''
        automatically run to extract just street coordinates from the openaddresses geojsons 
        '''
        for state in tqdm(self.states):
            
            print(f'extracting state {state}')
            all_jsons = glob.glob(os.path.join(state,"*.geojson"))
            coordinates = []
            for json_file in all_jsons:
                # ! parcels and buildings not relavant
                if not any(x in json_file for x in ['parcels','buildings']):
                    with open(json_file) as f:
                        for line in tqdm(f):
                            data = json.loads(line)
                            # ! Some dont have coordinates for whatever reason
                            if data['geometry'] is not None:
                                longitdue,latitude = data['geometry']['coordinates']
                                coordinates.append([latitude,longitdue])
                            else: 
                                print(f'{json_file} skipped')


            outdir = os.path.join(self.coords_dir,P(state).stem)+'.txt'

            with open(outdir,'w') as f:
                for coordinate in coordinates:
                    f.write(f'{coordinate[0]},{coordinate[1]}\n')
            f.close()

    def update_bad_indices(self,new_bad_indices):
        '''
        Depreciated. Was used to add bad indices while retriving data before, but that was kinda dumb.
        I can just check all the coordinates first.
        '''
        self.bad_indices = self.bad_indices.update(new_bad_indices)
        with open(self.bad_indices_dir,'a') as f:
            for idx in new_bad_indices:
                f.write(f'{idx}\n')
            print(f'{len(new_bad_indices)} new BAD indices added')
        f.close()


    def delete_all(self):
        '''
        removed all processed data/sampled indices. Mainly used when debugging, but can also be used
        to wipe everything for fresh processing.
        '''
        
        print("THIS WILL DELETE ALL LABEL FILES + SAMPLED INDICES")
        x = input('y/n?')

        if x.strip().lower() == 'y':
            files = glob.glob(os.path.join(self.label_dir,'*'))
            for f in files:
                if 'bad_indices' not in f:
                    print(f'removing {f}')
                    os.remove(f)
            files = glob.glob(os.path.join(self.data_root,'sampled_coordinates','*'))
            for f in files:
                print(f'removing {f}')
                os.remove(f)
        else:
            # lazy
            return


    def remove_all_bad_indices(self):
        bad_indices = []
        for name,state in tqdm(self.state_dict.items()):
            # state.update_sampled_indices(indices)
            coords = np.array(state.load_coordinates())
            for idx in range(self.max_dataset_size//len(self.state_dict) -1):
                # selected_coords = coords[idx]
                # for i in range(len(selected_coords)):          
                cur_coords = coords[idx].split(',')  
                if idx in bad_indices:
                    print(f'{idx} is bad, skipped')

                elif  self.xmin < float(cur_coords[0]) < self.xmax \
                    and self.ymin < float(cur_coords[1]) < self.ymax:
                    continue
                else:
                    bad_indices += [idx]
                    print(f'index {idx} for {name} is bad with coordinates: {coords[idx]}')
        # state.update_sampled_indices(indices_sampled)
        
        self.update_bad_indices(bad_indices)

            
    def select_indices(self,split_ratios):
        '''
        param: split_ratios: List[Int,Int,Int]
               number of train,val,test coords
        

        Given N target coordinates, sample N//51 from each state.
        -> rounded down, and coordinates with downloading errors not compensated for,
        -> so actual amount is going to be fewer
        '''
        if not isinstance(sum(split_ratios),int): raise TypeError('non integer in split ratios') 


        train,val,test = split_ratios
        N = len(self.state_dict)
        num_train = train// N
        num_val = val// N
        num_test = test// N 
        num_total = num_train+num_val+num_test # total number per state
        if num_total > self.max_dataset_size // N:
            raise AssertionError(f'split too large. Max dataset size is {self.max_dataset_size}, but {sum(split_ratios)} requested')


        print(f'After rounding, there will be {num_train*N} training samples')
        print(f'differnce of {train-num_train*N}')
        print(f'After rounding, there will be {num_val*N} validation samples')
        print(f'differnce of {val-num_val*N}')
        print(f'After rounding, there will be {num_test*N} test samples')
        print(f'differnce of {test-num_test*N}')
        print(f'new splits: {num_train*N},{num_val*N},{num_test*N}')
        print(f'old splits: {train},{val},{test}')
        

        #! EXCLUDE ALREADY SAMPLED INDICES AND BAD INDICES (download error)
        all_sampled_indices = [state.sampled_indices for _,state in self.state_dict.items()]
        all_sampled_indices = set(chain.from_iterable(all_sampled_indices))
        if all_sampled_indices is None:
            to_exclude = self.bad_indices
        else:
            to_exclude = all_sampled_indices.union(self.bad_indices)

        # all possible indices to sample from
        indices = np.array(list(range(0,self.max_dataset_size//N -1 )))
        to_exclude = np.array(list(to_exclude))

        # num_totals of valid randomly selected coordinates.
        indices = np.random.choice(indices[np.isin(indices,to_exclude,invert=True)],num_total,replace=False)

        train_indices = indices[:num_train]
        val_indices = indices[num_train:(num_train+num_val)]
        test_indices = indices[(num_train+num_val):]

        splits = {
            'train': train_indices,
            'val':val_indices,
            'test':test_indices
        }


        true_count = {
            'train': 0,
            'val':0,
            'test':0
        }

        for name,state in tqdm(self.state_dict.items()):

            coords = np.array(state.load_coordinates())
            indices_sampled = []
            for split,idx in splits.items():
                selected_coords = coords[idx]
                
                for i in tqdm(range(len(selected_coords))):
                    cur_coords = selected_coords[i].split(',')                    
                    if idx[i] in self.bad_indices:
                        print(f'{idx[i]} is bad, skipped')

                    elif  self.xmin < float(cur_coords[0]) < self.xmax \
                        and self.ymin < float(cur_coords[1]) < self.ymax:

                        outname = os.path.join(self.label_dir,split+'_coords.txt')
                        with open(outname,'a') as f:
                            f.write(f'{selected_coords[i]}\n')
                        f.close()
                        
                        outname = os.path.join(self.label_dir,'all_coords.txt')
                        with open(outname,'a') as f:
                            f.write(f'{selected_coords[i]}\n')
                        f.close()

                        outname = os.path.join(self.label_dir,split+'_state_idx.txt')
                        with open(outname,'a') as f:
                            f.write(f'{name}{idx[i]}\n')
                        f.close()

                        outname = os.path.join(self.label_dir,'all_state_idx.txt')
                        with open(outname,'a') as f:
                            f.write(f'{name}{idx[i]}\n')
                        f.close()
                        indices_sampled += [idx[i]]
                        true_count[split] += 1

                    else:
                        print(f'somehow coordinates are still bad: {cur_coords}')
            state.update_sampled_indices(indices_sampled)

        # print actual number sampled.
        print('TRUE SPLIT COUNTS:')
        for split,num in true_count.items():
            print(f'{split}:{num}')
        
    def concat_images(self,list_of_images,out_name):
        '''
        For every coordinate, we download 5 images at different angles and stitch them together
        '''
        images = []

        for img in list_of_images:
            cur_img = Image.open(img)
            images += [cur_img]
            x,y = cur_img.size

        new_image = Image.new("RGB", (x*len(list_of_images), y), "white")


        for i in range(len(images)):
            new_image.paste(images[i],(x*i,0))

        new_image.save(out_name)
        print(f'downloaded and done concat: {out_name}')


    def fetch_and_stitch(self,coordinate,split,stateidx):
        '''
        coordinate: [lat,long]
        split: str: 'train','val','test'
        stateidx: 'az003240'


        this function download and concats images for a single coordinate

        '''
        if self.API_KEY is None or self.SECRET is None:
            raise ValueError('API_KEY OR SECRET IS MISSING')

        split_folder = os.path.join(self.data_root,'images',split)
        temp_folder = os.path.join(self.data_root,'images','raw')
        final_name = os.path.join(split_folder,stateidx+'.jpg')
    
        if P(final_name).exists():
            print(f'{final_name} already exists! Skipping')
            return False
        
        headings = [0,72,72*2,72*3,72*4]
        images = []

        base_url = 'https://maps.googleapis.com/maps/api/streetview'
        
        for heading in headings:
            file_name = stateidx+'_'+str(heading).zfill(3)+'.jpg'
            params = urllib.parse.urlencode({
                'size': '448x448',
                'location': f'{coordinate[0]},{coordinate[1]}',
                'fov': '75',
                'heading': f'{heading}',
                'pitch': '0',
                'key': self.API_KEY,
                'return_error_code': 'true'
                })
            url = f"{base_url}?{params}"
            out_name = os.path.join(temp_folder,file_name)


            url = sign_url(input_url=url,secret=self.SECRET)

            try:
                response = urllib.request.urlretrieve(url,out_name)

            except urllib.error.URLError:
                print('error')
                with open(f'{self.data_root}/images/badcoordinates.txt','a') as f:
                    f.write(f'{stateidx}\n')
                    self.coordinates_with_errors += [stateidx]
                f.close()
                return False
            images += [response[0]]

        self.concat_images(images,final_name)
        return True

    def download_all_images(self,start_idx=0,num_frames=None):
        '''
        For every split, we try downloading the partitioned training/val/test coordinates
        if there is an error it skips the coordinate and appends it to bad_indices .txt
        
        '''        
        self.start_idx = start_idx
        self.num_frames = num_frames


        print(f'starting at frame_num: {start_idx}')
        state_idx = {
            'train':'',
            'val':'',
            'test':'',
        }

        all_coordinates = {
            'train':'',
            'val':'',
            'test':'',    
        }

        for key in state_idx:
            idx_file = self.label_dir+f'/{key}_state_idx.txt'
            coord_file = self.label_dir+f'/{key}_coords.txt'

            indices = [''.join(idx.split(',')) for idx in  open(idx_file).read().split('\n') if idx.strip() != ''] #maybe wrong
            state_idx[key] = indices

            coords = [c.split(',') for c in open(coord_file).read().split('\n') if c.strip() != '']
            coords = [[float(c[0]),float(c[1])] for c in coords]
            all_coordinates[key] = coords
        counter = 0
        
        for split in state_idx:

            names = state_idx[split]
            coordinates = all_coordinates[split]
            assert len(coordinates) == len(names), 'length mismatch'

            end_idx = len(coordinates) if self.num_frames is None else self.num_frames+start_idx
            
            pbar = tqdm(range(self.start_idx,end_idx))
            for i in pbar:
                
                if names[i] in self.coordinates_with_errors:
                    print(f'{names[i]} gives errors, skipping')
                    continue

                counter += self.fetch_and_stitch(coordinates[i],split,names[i]) * 5
                pbar.set_description("Successful images %s" % counter)