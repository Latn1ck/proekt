from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import chernovik
import chernovik2
import random


tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
def preprocess(x):
    x=x.replace('\n',' ')
    x=x.replace('\t','')
    return x[10:]
def embed_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()
def predict(x,vectClasses,norm):
    xVec=embed_bert(x)
    dists=np.array([norm(xVec-i) for i in vectClasses])
    return np.argmin(dists)+1
def accuracy(pred, ground_truth):
    return np.sum(pred==ground_truth)/len(pred)
""" vector1 = embed_bert("Кроссовки")
vector2=embed_bert("Ботинки")
vector3=embed_bert("Колбаса")
vector4=embed_bert("Кеды")
print(f'1,2: {np.linalg.norm(vector1-vector2)}')
print(f'1,3: {np.linalg.norm(vector1-vector3)}')
print(f'1,4: {np.linalg.norm(vector1-vector4)}')
 """

dictOKRB=chernovik.dictOKRBklass
shorthand={int(k):preprocess(v) for k, v in dictOKRB.items() if len(k)==2}
vectorizedClasses=list(shorthand.values())
vectorizedClasses=list(map(embed_bert,vectorizedClasses))
train_X=chernovik2.func
N=len(train_X)
print(N)
train_y=chernovik2.okrbLabels
print(len(train_y))
ix=random.sample(list(range(N)),100)
short_X=train_X[ix]
short_y=train_y[ix]
short_yPred=np.array([predict(i,vectorizedClasses,np.linalg.norm) for i in short_X])
print(accuracy(short_yPred,short_y))