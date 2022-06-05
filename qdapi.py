from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from pandas import read_csv, concat
from pandas import DataFrame
from keras.metrics import top_k_categorical_accuracy
from ast import literal_eval
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
import os
import psutil
import json
from collections import OrderedDict
import time

STROKE_COUNT = 500
animal_list = ['ant', 'bee', 'bird', 'camel', 'cat', 'cow', 'crab', 'crocodile', 'dolphin', 'elephant', 'fish',
               'giraffe', 'horse', 'kangaroo', 'lion', 'monkey', 'mouse', 'octopus', 'owl', 'panda', 'penguin', 'pig',
               'rabbit', 'sea turtle', 'shark', 'sheep', 'snail', 'snake', 'squirrel', 'swan', 'tiger', 'whale',
               'zebra', 'circle', 'flower', 'sun', 'triangle']

word_encoder = LabelEncoder()
word_encoder.fit(animal_list)


def top_3_accuracy(x, y):
    return top_k_categorical_accuracy(x, y, 3)


def _stack_it(raw_strokes):
    """preprocess the string and make
    a standard Nx3 stroke vector"""
    # stroke_vec = literal_eval(raw_strokes)  # string->list
    # unwrap the list
    stroke_vec = raw_strokes
    in_strokes = [(xi, yi, i) for i, (x, y, t) in enumerate(stroke_vec) for xi, yi in zip(x, y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:, 2] = [1] + np.diff(c_strokes[:, 2]).tolist()
    c_strokes[:, 2] += 1  # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1),
                         maxlen=STROKE_COUNT,
                         padding='post').swapaxes(0, 1)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def get_result():
    # global predicted_result
    print(1111)
    if request.method == 'POST':
        json_data = request.get_json()
        # with open(jsonfile) as json_file:
        #     json_data = json.load(json_file)
        json_end = json_data["isEnd"]
        json_key = json_data["key_id"]
        json_drawing = json_data["drawing"]
        json_solving = json_data["playing_word"]

        if json_end:
            # json_drawing 편집 후에 key 랑 이것저것 해서 dataframe 생성,
            # 저장하기
            tf = json_end - 1
            df = DataFrame({'key_id': json_key, 'word': json_solving, 'recognized': tf, 'countrycode': ['KOR'],
                            'drawing': json_drawing})
            try:
                user_history = read_csv("./userdrawing/{}.csv".format(json_key))
                result = concat([user_history, df])
            except:
                result = df

            result.to_csv("./userdrawing/{}.csv".format(json_key))
            return {'return': 'end'}

        else:
            #     drawing 형식이 같다고 생각하고 코드 진행
            # json_drawing = json_drawing.map(_stack_it)
            # json_drawing = literal_eval(json_drawing)
            print('1', type(json_drawing), json_drawing)

            json_drawing = _stack_it(json_drawing)
            sub_vec = np.stack(json_drawing, 0)  # json_drawing.values
            # sub_vec = sub_vec.tolist()
            sub_vec = np.array(sub_vec, ndmin=3)
            sub_pred = model.predict(sub_vec, verbose=True, batch_size=4096)
            predicted_result = [word_encoder.classes_[np.argsort(-1 * sub_pred)[0:1]]]
            predicted_result = predicted_result[0][0][0]
            # predicted_data = OrderedDict()
            # predicted_data['predicted_animal'] = predicted_result
            # with open('predicted_data.json', 'w', encoding='utf-8') as make_file:
            #     json.dump(predicted_data, make_file, ensure_ascii=False, indent="\t")

            return {'predicted_animal': predicted_result}


if __name__ == '__main__':
    model = load_model('0805.h5', custom_objects={"top_3_accuracy": top_3_accuracy})
    model._make_predict_function()
    app.run(host='0.0.0.0', port=5000, debug=True)