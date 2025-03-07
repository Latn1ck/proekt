import chernovik as ch
import pandas as pd
import re
import numpy as np

def getChapter(x):
    if pd.isna(x):
        return '00'
    x=re.sub(r'[а-яА-я\s]','',x)
    if len(x)==1:
        return '00'
    if x[1] in ['.',',']:
        return '0'+x[0]
    return x[:2]

df=pd.read_csv('parsed_tradeitem.csv',sep=';')
print(list(set(df['Ticountryoflastprocessing'])))
nans=pd.DataFrame(df.isna().sum())
nans=nans.rename(columns={0:'score'})
nans['Freq']=nans['score']/df.shape[0]
nans=nans.T
threshold=0.9
vectorDim=10
train=df.copy()
okrbLabels=train['Okrb007']
okrbLabels=okrbLabels.apply(getChapter)
print(okrbLabels)
bricks=train['GpcBrick']
gpcLabels=train[['GpcSegm','GpcFamily','GpcClass']]
tnvedLabels=train[['Tnvedcode','Tnvedpath']]
train=train.drop(columns=list(nans.loc[:, nans.loc['Freq']>threshold].columns)+['Okrb007','GpcBrick','GpcSegm','GpcFamily','GpcClass','Tnvedcode','Tnvedpath'])  
train=train.sample(n=50000, random_state=42)
