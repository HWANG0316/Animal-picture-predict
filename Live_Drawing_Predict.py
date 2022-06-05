import matplotlib
import pandas as pd
from pandas import DataFrame
import time
matplotlib.use('Agg')
import tkinter.font
from tkinter import *
import os
import sys
import numpy as np
from numpy import random
from keras.models import load_model
from ast import literal_eval
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y, 3)

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

STROKE_COUNT = 500 #최대로 용인되는 그림당 스트로크 수

#참고용~
import tensorflow as tf
tf.__version__
import keras
keras.__version__

import sys
print(sys.version)

def _stack_it(raw_strokes):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    stroke_vec = literal_eval(raw_strokes) # string->list
    stroke_vec = raw_strokes
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
    


# 게임에 나타날 동물 리스트를 모두 기입해주세요
animal_list = ['ant', 'bee' ,'bird' ,'camel' ,'cat' ,'cow' ,'crab' ,'crocodile' ,'dolphin' ,'elephant' ,'fish' ,'giraffe' ,'horse' ,'kangaroo' ,'lion','monkey','mouse','octopus' ,'owl' ,'panda' ,'penguin' ,'pig' ,'rabbit' ,'sea turtle' ,'shark','sheep','snail' ,'snake' ,'squirrel' ,'swan' ,'tiger' ,'whale' ,'zebra', 'circle' , 'flower', 'sun', 'triangle' ]

# #model import 시, 오류를 막기 위한 라벨 인코더 인코딩
word_encoder = LabelEncoder()
word_encoder.fit(animal_list)
# print('words', len(word_encoder.classes_), '=>', ', '.join([x for x in word_encoder.classes_]))


#변환된 CSV 파일을 저장할 이름을 설정해주세요
save_filename="0821.csv"

#predict에 사용할 모델을 불러오세여 
loaded_model= load_model("0805.h5",custom_objects={"top_3_accuracy": top_3_accuracy})
loaded_model.summary()
# 한번에 변환하고자 하는 파일 수를 전역변수로 설정해준다.
global Play
played_num=0
global t
t=0

# Drawing을 위한 Canvas의 크기를 설정해준다.
canvas_width = 1000
canvas_height = 600

#각 스트로크 사이를 분리하여 저장하도록 도와주는 메서드이다. 
def cutter(cut_time,strokes):
    indexlist=[] #indexlist는 마우스를 떼는 순간의 좌표를 기록할 list이다.
    for j in range(len(cut_time[0])-1):
        xx=cut_time[0][j]  #cut_time은 마우스를 뗀 순간의 시점을 기록한 list이다. 아래 블럭을 참고한다면 이해할 수 있다. 
        yy=cut_time[1][j]
        xindex=[i for i,x in enumerate(strokes[0][0][0]) if x==xx]
        yindex=[i for i,y in enumerate(strokes[0][0][1]) if y==yy]
        indexlist.append(list(set(xindex).intersection(yindex)))


    for i in range(len(indexlist)):
        indexlist[i]=indexlist[i][0]
        
        
    for i in range(len(indexlist)):
        indexlist[i]=indexlist[i]+1

    accepter=[[]]
    strokes=np.array(strokes)
        
    # Stroke 분할
    for i in range(len(indexlist)+1):
        accepter[0].append(((np.hsplit(strokes[0][0],indexlist))[i]).tolist())
    strokes.tolist()
    return accepter

#remove 메서드는 캔버스 프로토타입에서 지우기 버튼을 눌렀을 때의 상황을 처리하는 기능을 가졌다. 
def remove(canvas,strokes,cut_time):
    canvas.delete("all")
    strokes[:]=[[[[],[],[]]]]
    cut_time[:]=[[],[]]
    predicted_animal.set("지금 그림을 지운거에요??!")
    
    
### Button1 click event를 기준으로 폭이 1인 드로잉 툴을 사용
def paint( event ):
    global x0,y0
    python_green = "#476042"    # drawing pen의 color
    strokes[0][0][0].append(int(event.x))                
    strokes[0][0][1].append(int(event.y))
    strokes[0][0][2].append(time.perf_counter())
    w.create_line(x0,y0,event.x,event.y,width=3.5)
    x0, y0 = event.x, event.y
    time.sleep(0.02)

#### (Optional) oval tool로 drawing하려면 아래의 코드를 paint 함수에 넣기
#     x1, y1 = ( event.x - 1 ), ( event.y - 1 )
#     x2, y2 = ( event.x + 1 ), ( event.y + 1 )
#     w.create_oval( x1, y1, x2, y2, fill = python_green )

#button1을 클릭하는 순간 실행되는 메서드
def down(event):
    global x0,y0
    x0, y0= event.x, event.y
    
### stroke split을 위한 checkpoint를 Button1-release로 기록
def splitcheck( event ):
    cut_time[0].append(event.x)
    cut_time[1].append(event.y)
    global x0, y0
    if(x0,y0)==(event.x, event.y):
        w.create_line(x0,y0,x0+1,y0+1)
    cut_result = cutter(cut_time,strokes)
#     print('cut result:' , cut_result[0] ) ## 출력 대신에 여기서 predict가 실시간으로 이루어져야 한다.
    translater=DataFrame({'drawing' : cut_result})

    translater['drawing']=translater['drawing'].map(_stack_it)
    sub_vec = np.stack(translater['drawing'].values, 0)

#     cut_result[0]=cut_result[0].map(_stack_it)
#     sub_vec= np.stack ( cut_result[0].values , 0 )
    sub_pred = loaded_model.predict(sub_vec, verbose=True, batch_size=4096) #실시간 예측을 위한 코드
    predicted_result= [word_encoder.classes_[np.argsort(-1*sub_pred)[0:1]]]
    predicted_result= predicted_result[0][0][0]
    now_animal=draw_animal[Play-1]
    if now_animal == predicted_result:
        predicted_animal.set("아 알겠어요! 이거 {} 맞죠?".format(predicted_result))
    else :
        predicted_animal.set("혹시 {} 를 그린건가요?!?!?!".format(predicted_result))
    time.sleep(0.015) # stroke간의 간격이 너무 좁아지는 것을 방지
    
    
    
    global Game
Game = 1   ## Game 변수는 반복문을 수행하도록 하는 boolean 변수의 기능을 한다
Play = 3   ## Play 변수는 게임 한 판 동안 몇 번의 드로잉을 진행할 지 정해준다


    
while(Game):
    draw_animal=[]    #draw 할 동물을 list로 만든다
    rand = np.random.randint(0,len(animal_list),Play)   #랜덤으로 Play 수만큼 동물을 추출한다
    print(rand)

    for randomnum in rand:
        draw_animal.append(animal_list[randomnum])
        print(randomnum)
    
        
    while(Play):  
        strokes=[[[[],[],[]]]]
        cut_time=[[],[]]
        master = Tk()  # Tk 객체 생성

        master.title( "NOW PLAYING : GuickDraw" )
        master.geometry('1400x800+0+0')
        master.resizable(False, False)
        w = Canvas(master,
                   relief='solid',
                   bd=3,
                   bg='white',
                   width=canvas_width, 
                   height=canvas_height
                   )
        font1=tkinter.font.Font(family='나눔스퀘어라운드 Regular',size=14)
        font2=tkinter.font.Font(family='나눔스퀘어라운드 Regular',size=30)
        font3=tkinter.font.Font(family='나눔스퀘어라운드 Regular',size=20)
        label1=Label(master, text = '아래 캔버스에 다음 동물을 그리세요', bg='#FFD000',font=font1, padx=640, pady=5)
        label2=Label(master,text = draw_animal[Play-1], padx=662, bg="#FFD000",font=font2, pady=3)
        label3=Label(master, text='              지금 {} 번째 드로잉중!    목표 : {} 개'.format(played_num+1,played_num+Play), bg="#FFD000", font=font1, padx=658, pady=1)

        predicted_animal=StringVar()    ## 스트로크별로 예측된 결과를 StringVar 변수에 저장한다
        if cut_time==[[],[]] :
            predicted_animal.set("아직 암것도 안그리셨눼 ~,~ ")

        label4=Label(master, font= font1, textvariable = predicted_animal)
        if(predicted_animal == "아 알겠어요! 이거 {} 맞죠?".format(draw_animal[Play-1])):
            time.sleep(1)
            master.destroy()
        

        btn1=Button(master, text="지우고 다시 그릴래요~", bg='#BBEEAA',padx=850,pady=10,font=font3,command= lambda:remove(w, strokes, cut_time))
        btn2=Button(master, text="다음 문제로!",bg='#FAC0D0', padx=120, pady=200, font=font3,command= lambda: master.destroy())


        label1.pack(side='top')
        label2.pack(side='top')
        label4.pack()
        label3.pack()

        btn2.pack(side='right')
        btn1.pack()

        w.pack(side='top') 


        w.bind( "<B1-Motion>", paint )
        w.bind( "<ButtonRelease-1>", splitcheck)
        w.bind("<Button-1>", down)
    #     message = Label( master, text = "Press and Drag the mouse to draw" )
    #     message.pack( side = BOTTOM )
        mainloop()
        




        # print('strokes: ',type(strokes),strokes )
        # print('cut_time: ', type(cut_time),cut_time)


        ## stroke split 
        indexlist=[]
        for j in range(len(cut_time[0])-1):
            xx=cut_time[0][j]
            yy=cut_time[1][j]
            xindex=[i for i,x in enumerate(strokes[0][0][0]) if x==xx]
            yindex=[i for i,y in enumerate(strokes[0][0][1]) if y==yy]
            indexlist.append(list(set(xindex).intersection(yindex)))



        try:
            for i in range(len(indexlist)):
                indexlist[i]=indexlist[i][0]



            for i in range(len(indexlist)):
                indexlist[i]=indexlist[i]+1

            accepter=[[]]
            strokes=np.array(strokes)

            # Stroke 분할
            for i in range(len(indexlist)+1):
                accepter[0].append(((np.hsplit(strokes[0][0],indexlist))[i]).tolist())


            # 각 개체에 고유 Key 배당
            from random import *
            id=randint(1,1000000)

            df=DataFrame({'key_id' : id , 'word':draw_animal[Play-1], 'countrycode' : ['KOR'],  'drawing' : accepter})

            ## 아래 Try-Except 문은 기존의 파일에 새로 생성한 drawing 개체를 병합하는 과정을 포함하고 있다.
            try :
                result=pd.read_csv(save_filename)
                frames=[result, df]
                result=pd.concat(frames)
            except:
                result=df


            result.to_csv(save_filename,index=False)
            Play=Play-1
            played_num += 1
            

        except IndexError : 
            Play=Play
            played_num=played_num



    if(Play == 0):
        master=Tk()
        master.title("END!")
        master.geometry('1400x800+0+0')
        master.resizable(False, False)
        font2=tkinter.font.Font(family='나눔스퀘어라운드 Regular',size=30)
        Endlabel=Label(master, text='끝났슈',padx=100, pady=50, font=font2).pack(side='top')
        def endgame(master):
            global Game
            Game=0
            master.destroy()

        regamebutton=Button(text='한번 더!',padx=500, pady=100, font=font2,bg="#FFD000", command=lambda : replay(master)).pack(side='top')
        quitbutton=Button(text='이제 그만할래요',padx=500, pady=80, font=font2 ,bg="#CCC0D0",fg="black", command=lambda : endgame(master)).pack(side='top')
        def replay(master):
            global Play
            Play=3
            master.destroy()
            
       

        mainloop()

    