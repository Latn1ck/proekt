import fasttext
from datetime import datetime
from hiclass.metrics import f1,precision,recall
from hiclass import LocalClassifierPerParentNode
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd


model=fasttext.load_model('new.bin')
X_train=[]
y_train=[]
with open('test.txt','r',encoding='utf-8') as f:
    for line in f:
        line=line.split()
        y_train.append((line[2])[-6:])
        X_train.append(model.get_word_vector(' '.join(line[3:])))
X_test=[]
y_test=[]
with open('test.txt','r',encoding='utf-8') as f:
    for line in f:
        line=line.split()
        y_test.append((line[2])[-6:])
        X_test.append(model.get_word_vector(' '.join(line[3:])))

local=LogisticRegression(max_iter=1000,random_state=42)
print(f'start {datetime.now()}')
clf=joblib.load('clf.joblib')
y_pred=list(map(lambda x:''.join(x),clf.predict(X_test)))
print(f'f1={f1(y_pred,y_test)}')#0.9198958083822341
print(f'precision={precision(y_pred,y_test)}')#0.9214284758866529
print(f'recall={recall(y_pred,y_test)}')#0.9183682311672513
print(f'finish {datetime.now()}')