import pandas as pd
from pandas import DataFrame
from IPython.display import clear_output
import os
from glob import glob
import time
import tensorflow.compat.v1 as tf

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


tf.disable_v2_behavior()

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

base_dir=os.path.join('./','ndjsonlist')
ndjson_list=glob(os.path.join(base_dir,'*.ndjson'))
ndjson_list.index("./ndjsonlist/triangle.ndjson")
ndjson_list=ndjson_list[13:]
print(ndjson_list)


# base_dir=os.path.join('./','ndjsonlist')
# ndjson_list=glob(os.path.join(base_dir,'*.ndjson'))

for ndjson in ndjson_list:
    with open(ndjson,'r') as f:
        data=f.readlines()
    for i, datum in enumerate(data):
        
        ind1=ndjson.find('dataset')+14
        ind2=ndjson.find('.ndjson')
        animalname=ndjson[ind1:ind2]
        print("animalname:", animalname)
        if(datum.find("true") != -1):
            key_index1=datum.find("key_id")+9
            key_index2=datum.find("drawing")-3
            key=datum[key_index1:key_index2]
            drawing_index1=datum.find('[[[')
            drawing_index2=datum.find(']]]')+3
            drawing=datum[drawing_index1:drawing_index2]
            new_df=DataFrame({"word":[animalname], "countrycode":["CSLEE"], "timestamp":["Current"], "recognized": ["true"] , "key_id": [key] , "drawing":[drawing]})
            try:
                frames=[df,new_df]
                df=pd.concat(frames)
            except NameError: df=new_df
            
        clear_output(wait=True)
        print('%.2f  %%  /  100 %%' %(100*i/len(data)))
        

    df.to_csv("./ndjsonlist/{}.csv".format(animalname),index= False)
    print(animalname)
    time.sleep(1)