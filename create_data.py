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
from data_utils import Datatools
from creds import APIKEY1,APIKEY2

tools = Datatools(
                data_root = '/home/gabriel/guesser/data',          
                raw_json_dirs='raw_jsons/us',
                coords_dir='coordinates_by_state')

# tools.extract_coordinates()
# tools.remove_all_bad_indices()
# tools.delete_all()
# concat_images(directory='/home/gabriel/guesser/test_images')
# tools.select_indices([42000,4000,4000])

APIKEY = APIKEY1
tools.API_KEY = APIKEY['key']
tools.SECRET = APIKEY['secret']
tools.download_all_images()
# tools.fetch_and_stitch([44.2778261,-105.5066049],'train','ak123201')



# key1_start = 13788 
# end_1/start2= 19493 
# ened 2 = 25198
# remaining = 11410
# remaining/2 = 5705


# key2_start = 25929+971 



# remaining = 14200
# remaining/2 = 7100






# 
# img = Image.open("logo.png")
# img1 = Image.open("logo2.png")
# tools._count_coordinates()

# az = State('data/coordinates_by_state/fl.txt','az')
# print(len(az))

# az.test = 52
# print(az.path)
# print(az.test)    

# root_dir = 'data/raw_jsons/us'
# out_path = 'data/coordinates_by_state'

# states = glob.glob(os.path.join(root_dir,"*/"))


# for state in states:
#     # data/locations/us/state    
#     print(f'extracting state {state}')
#     all_jsons = glob.glob(os.path.join(state,"*.geojson"))
#     coordinates = []
#     for json_file in all_jsons:
#         if not any(x in json_file for x in ['parcels','buildings']):
#             with open(json_file) as f:
                
#                 for line in tqdm(f):
#                     data = json.loads(line)

#                     if data['geometry'] is not None:
#                         longitdue,latitude = data['geometry']['coordinates']
#                         coordinates.append([latitude,longitdue])
#                     else: 
#                         print(f'{json_file} skipped')


#     outdir = os.path.join(out_path,P(state).stem)+'.txt'

#     with open(outdir,'w') as f:
#         for coordinate in coordinates:
#             f.write(f'{coordinate[0]},{coordinate[1]}\n')
#     f.close()
#     # break






# def sample_locations(train_val_test_ratio,coordinates_by_state):
#     all_states = glob.glob(os.path.join(coordinates_by_state,"/*.txt"))

#     train,val,test = [],[],[]
#     # for 
#     with open('C:/path/numbers.txt') as f:
#         lines = f.read().splitlines()







# # %%|
# test_coordinate = None
# # %%
# API = 'AIzaSyCTAUksORxU8pRQ7RHBxirSoIeLj01XoqU'



# params = urllib.parse.urlencode({
#     'key': API,
#     'size': '448x448',
#     'location': f'{test_coordinate[0]},{test_coordinate[1]}',
#     'heading': f'{0}',
#     'pitch': '0',
#     'fov': '75',
#     'return_error_code': 'true'
#     })

# base_url = 'https://maps.googleapis.com/maps/api/streetview'


# url = f"{base_url}?{params}"

# # %%
# url
# # %%
# try:
#     response = urllib.request.urlretrieve(url,"00000001.jpg")

# except urllib.error.URLError:
#     print('error')


# # %%
# response
# # %%
