batch_size = 128 #batch! batch!
STROKE_COUNT = 500 #limit of stroke(한 스트로크에 담을 수 있는 최대 point 개수)
TRAIN_SAMPLES = 750 
VALID_SAMPLES = 75
TEST_SAMPLES = 50


%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pandas import DataFrame 
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y, 3)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from glob import glob
import psutil
import gc
gc.enable()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

base_dir = './dataset2'
# base_dir = './QuickDraw/dataset' # model train 에 사용될 csv data들이 들어있는 폴더 디렉토리



from ast import literal_eval
# string => list 의 기능을 해준다

ALL_TRAIN_PATHS = glob(os.path.join(base_dir, '*.csv'))
print(ALL_TRAIN_PATHS)
COL_NAMES = ['draw_animal', 'drawing', 'company_cd', 'age_group'] 

#stack it 함수는 말 그대로 쌓아주는 함수이다.
#[[x1,x2,x3,x4,,,],[y1,y2,y3,y4...],[z1,z2,z3,z4,,,,,]]의 형태를 [[x1,y1,z1],[x2,y2,z2],,,,]의 꼴로 바꿔줌
def _stack_it(raw_strokes):   
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    stroke_vec = literal_eval(raw_strokes) # string->list
    
    # unwrap the list
    in_strokes = [(xi,yi,i) for i,(x,y,t) in enumerate(stroke_vec) for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1), 
                         maxlen= STROKE_COUNT, 
                         padding='post').swapaxes(0, 1)


def read_batch(samples=5, 
               start_row=0,
               max_rows = 1000):
    """
    load and process the csv files
    this function is horribly inefficient but simple
    """
    out_df_list = []
    for c_path in ALL_TRAIN_PATHS:
        c_df = pd.read_csv(c_path, nrows=max_rows, skiprows=start_row)
        c_df.columns=COL_NAMES
        out_df_list += [c_df.sample(samples)[['draw_animal', 'drawing' , 'company_cd']]]
    full_df = pd.concat(out_df_list)
    full_df['drawing'] = full_df['drawing'].map(_stack_it)
    
    return full_df

## QuickDraw! 에서 제공된 Dataset 중에서 True로 tag된 data만 사용함(정확도 개선을 위함)
train_args = dict(samples=TRAIN_SAMPLES, 
                  start_row=0, 
                  max_rows=int(TRAIN_SAMPLES*1.5))
valid_args = dict(samples=VALID_SAMPLES, 
                  start_row=train_args['max_rows']+1, 
                  max_rows=VALID_SAMPLES+25)
test_args = dict(samples=TEST_SAMPLES, 
                 start_row=valid_args['max_rows']+train_args['max_rows']+1, 
                 max_rows=TEST_SAMPLES+25)
print(1)

###dataset을 잘 보면 "recognized"라는 컬럼이 있는데, 구글이 해당 그림 데이터를 인식에 성공했는지 아닌지에 대한 정보로 추정됨. 
###해당 프로젝트에서는 True tag만 선별하여 학습 진행 --> 정확도 업!
###아래 19, 24, 29 줄이 이와 관련된 코드이다. 

#train dataset
train_df = read_batch(**train_args)
train_df = train_df[train_df.company_cd !=  False ] 
train_df = train_df.drop(['company_cd'],axis=1)
print(2)
#validation dataset
valid_df = read_batch(**valid_args)
valid_df = valid_df[valid_df.company_cd !=  False ]
valid_df = valid_df.drop(['company_cd'],axis=1)
print(3)
#test dataset
test_df = read_batch(**test_args)
test_df = test_df[test_df.company_cd !=  False ]
test_df = test_df.drop(['company_cd'],axis=1)

#LabelEncoder를 통하여 우리가 가지고있는 dataset의 동물종류를 encoding한다. 
word_encoder = LabelEncoder() 
word_encoder.fit(train_df['draw_animal'])
print()
print('draw_animals', len(word_encoder.classes_), '=>', ', '.join([x for x in word_encoder.classes_]))



# dataset에서 X에는 stroke 정보를, y에는 동물이름 정보를 one-hot-encoding으로 저장
def get_Xy(in_df):
    X = np.stack(in_df['drawing'], 0)
    y = to_categorical(word_encoder.transform(in_df['draw_animal'].values))
    return X, y
train_X, train_y = get_Xy(train_df)
valid_X, valid_y = get_Xy(valid_df)
test_X, test_y = get_Xy(test_df)
print(train_X.shape)

#여기는 그냥 dataset을 그림으로 한번 시험삼아 출력해보는 코드에용 
fig, m_axs = plt.subplots(3,3, figsize = (16, 16))
rand_idxs = np.random.choice(range(train_X.shape[0]), size = 9)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = train_X[c_id]
    test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points
    lab_idx = np.cumsum(test_arr[:,2]-1)
    for i in np.unique(lab_idx):
        c_ax.plot(test_arr[lab_idx==i,0], 
                np.max(test_arr[:,1])-test_arr[lab_idx==i,1], 'k')
    c_ax.axis('off')
    c_ax.set_title(word_encoder.classes_[np.argmax(train_y[c_id])])
    

# 아래의 긴 #라인의 부분은 gpu를 사용하여 학습하도록 하는 코드이므로, gpu를 사용하고자 한다면 주석을 해제하기만 하면 된다.
# gpu를 사용하면 좋겠지만, 사용하지 않았던 이유가 따로 있다. 이와 관련된 자세한 내용은 인수인계 문서 참조 바람.

################################################################################
# gpu를 인식하는지 확인
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
###############################################################################



from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
from keras.layers import LSTM
#############################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf. config.experimental.list_logical_devices('GPU')
        print(len(gpus),"Physical GPUs", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

if len(get_available_gpus())>0:
    # https://twitter.com/fchollet/status/918170264608817152?lang=en
    from keras.layers import CuDNNLSTM as LSTM 
    print(1)
    # this one is about 3x faster on GPU instances
#################################################################################
from keras.models import load_model

# model layer 구축
loaded_model=load_model("0805.h5",custom_objects={"top_3_accuracy": top_3_accuracy})
loaded_model.compile( loss = 'categorical_crossentropy',
                           optimizer = 'adam',  
                           metrics = ['categorical_accuracy', top_3_accuracy])
loaded_model.summary()
#loss, optimizer, metrics 설정하여 compile
#top_3_accuracy는 위쪽에서 따로 정의한 메서드다.
#실행시키면 나오는 무수한 WARNING들은 tensorflow 버전때문에 나오는 것이니 너무 걱정하지 않아도 된다.
#현재 이 모델은 tensorflow 1.15.0, keras 2.2.4 버전으로 작성되었다. 



# weight들을 저장해두는 path
weight_path="{}_weights.best2.hdf5".format('stroke_lstm_model')

#말그대로 checkpoint
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

#val_loss로 monitor하면서 조정하겠다는 의미 
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, 
                                   verbose=1, mode='auto', delta=0.0001, cooldown=5, min_lr=0.0001)
#patience 횟수의 epoch를 지나는 동안 val_loss의 감소효과가 연달아 안나타나면 학습을 중단시킨다. 
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=12) # Kaggle에서는 성능을 고려하여 patience=5로 설정하였으나, nipa서버에서는 12로
#callback 
callbacks_list = [checkpoint, reduceLROnPlat, early]



loaded_model.fit(train_X, 
                      train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = batch_size,
                      epochs = 200, #EarlyStopping Callback 때문에 어차피 중간에 끝나기 때문에 Epoch를 여유롭게 설정함
                      callbacks = callbacks_list,
                      )
#clear_output()



####################################################################################
# E tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
# GPU 사용 메모리 증가 허용으로 해결
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=confiig)
####################################################################################

print("Train_X:",train_X,"\nTrain_y",train_y,"\nvalid_x",valid_X,"\nvalid_y",valid_y)


#test set으로 train된 모델을 test한 결과 출력
loaded_model.load_weights(weight_path)
lstm_results = loaded_model.evaluate(test_X, test_y, batch_size = 4096)
print('Accuracy: %2.1f%%, Top 3 Accuracy %2.1f%%' % (100*lstm_results[1], 100*lstm_results[2]))



#학습된 모델을 hdf5 형식으로 save한다.
loaded_model.save('1119.h5')
#save한 모델을 다시 load하는 방법
#이 코드에서 의미있는 부분은 아니지만, 다른 곳에서 load하는 과정을 연습해보고자 넣어봤슴메
loaded_model=load_model("0805.h5",custom_objects={"top_3_accuracy": top_3_accuracy})
print('1', psutil.cpu_percent(interval=None))
loaded_model.summary()
print('2', psutil.cpu_percent(interval=None))


#그린 그림에 가까운 순서대로 top3 결과를 저장한 리스트

animal_list = ['ant', 'bee' ,'bird' ,'camel' ,'cat' ,'cow' ,'crab' ,'crocodile' ,'dolphin' ,'elephant' ,'fish' ,'giraffe' ,'horse' ,'kangaroo' ,'lion','monkey','mouse','octopus' ,'owl' ,'panda' ,'penguin' ,'pig' ,'rabbit' ,'sea turtle' ,'shark','sheep','snail' ,'snake' ,'squirrel' ,'swan' ,'tiger' ,'whale' ,'zebra', 'circle' , 'flower', 'sun', 'triangle' ]

# #model import 시, 오류를 막기 위한 라벨 인코더 인코딩
word_encoder = LabelEncoder()
word_encoder.fit(animal_list)

top_3_pred = [word_encoder.classes_[np.argsort(-1*c_pred)[0:3]] for c_pred in sub_pred]
top_3_pred = [' '.join([col.replace(' ', '_') for col in row]) for row in top_3_pred]
top_3_pred[0:3] #앞에서부터 세 개의 그림을 예측하여 각 그림의 top 3를 출력하는 코드

fig, m_axs = plt.subplots(2,4, figsize = (14, 14))
rand_idxs = np.random.choice(range(sub_vec.shape[0]), size = 25)
rand_idxs = range(0,16)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = sub_vec[c_id]
    test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points
    lab_idx = np.cumsum(test_arr[:,2]-1)
    for i in np.unique(lab_idx):
        c_ax.plot(test_arr[lab_idx==i,0], 
                np.max(test_arr[:,1])-test_arr[lab_idx==i,1], 'k')
    c_ax.axis('off')


sub_df['word'] = top_3_pred
sub_df[['key_id', 'word']].to_csv('submission.csv', index=False)
