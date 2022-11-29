from utils.data_utils import Datatools
from utils.creds import APIKEY1,APIKEY2

data_root = '/home/gabriel/guesser/data'
raw_json_dirs='raw_jsons/us'
coords_dir='coordinates_by_state'


tools = Datatools(
                data_root = data_root,          
                raw_json_dirs=raw_json_dirs,
                coords_dir=coords_dir)

tools.extract_coordinates()
tools.remove_all_bad_indices()
tools.select_indices([42000,4000,4000])

# TODO: create a thread for each api key to split up image downloading 
# TODO: with free GCP acc it downloads ~1 image/sec
APIKEY = APIKEY1
tools.API_KEY = APIKEY['key']
tools.SECRET = APIKEY['secret']
tools.download_all_images()
