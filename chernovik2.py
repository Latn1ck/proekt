import pandas as pd
import re
import numpy as np
import torch.nn as nn


def getChapter(x):
    if pd.isna(x):
        return 0
    x=re.sub(r'[а-яА-я\s]','',x)
    if len(x)==1:
        return 0
    if x[1] in ['.',',']:
        return int(x[0])
    return int(x[:2])

df=pd.read_csv('parsed_tradeitem.csv',sep=';',quotechar='"', skipinitialspace=True)
kusok=df.head(100000)
nans=pd.DataFrame(kusok.isna().sum())
nans=nans.rename(columns={0:'score'})
nans['Freq']=nans['score']/kusok.shape[0]
nans=nans.T
threshold=0.9
train=kusok.copy()
bricks=train['GpcBrick']
gpcLabels=train[['GpcSegm','GpcFamily','GpcClass']]
tnvedLabels=train[['Tnvedcode','Tnvedpath']]
classColumns=['Okrb007Path','Okrb007','GpcBrick','GpcSegm','GpcFamily','GpcClass','Tnvedcode','Tnvedpath']
okrbLabels=train['Okrb007']
okrbLabels=np.array(okrbLabels)
train=train.sample(n=50000, random_state=42)
X_train=np.array(train['Functionalname'])
y_train=np.array(train['Okrb007'].apply(getChapter))
test=df.iloc[100001:120001]
test=test.sample(n=5000,random_state=42)
X_test=np.array(test['Functionalname'])
y_test=np.array(test['Okrb007'].apply(getChapter))