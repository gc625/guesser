# %%
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
# %%

root_dir = 'data/raw_jsons/us'
out_path = 'data/coordinates_by_state'

states = glob.glob(os.path.join(root_dir,"*/"))


for state in states:
    # data/locations/us/state    
    
    all_jsons = glob.glob(os.path.join(state,"*.geojson"))
    coordinates = []
    for json_file in all_jsons:
        if not any(x in json_file for x in ['parcels','buildings']):
            with open(json_file) as f:
                
                for line in tqdm(f):
                    data = json.loads(line)
                    longitdue,latitude = data['geometry']['coordinates']
                    coordinates.append([latitude,longitdue])



    outdir = os.path.join(out_path,P(state).stem)+'.txt'

    with open(outdir,'w') as f:
        for coordinate in coordinates:
            f.write(f'{coordinate[0]},{coordinate[1]}\n')
    f.close()
    # break

# %%





def sample_locations(train_val_test_ratio,coordinates_by_state):
    all_states = glob.glob(os.path.join(coordinates_by_state,"/*.txt"))

    train,val,test = [],[],[]
    for 
    with open('C:/path/numbers.txt') as f:
        lines = f.read().splitlines()







# %%|
test_coordinate = None
# %%
API = 'AIzaSyCTAUksORxU8pRQ7RHBxirSoIeLj01XoqU'

params = urllib.parse.urlencode({
    'key': API,
    'size': '640x640',
    'location': f'{test_coordinate[0]},{test_coordinate[1]}',
    'heading': f'{0}',
    'pitch': '20',
    'fov': '90',
    'return_error_code': 'true'
    })

base_url = 'https://maps.googleapis.com/maps/api/streetview'


url = f"{base_url}?{params}"

# %%
url
# %%
try:
    response = urllib.request.urlretrieve(url,"00000001.jpg")

except urllib.error.URLError:
    print('error')


# %%
response
# %%
