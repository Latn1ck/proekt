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
def countryToInt(x):
    if pd.isna(x):
        return 0
    return int(x) if x.isdigit() else int(re.search(r"\[(\d+)\]", x).group(1))
def f(x):
    if pd.isna(x):
        return ''
    return re.search(r"\[(.*?)\]", x).group(1)
def count(x):
    return len(list(set(x)))
df=pd.read_csv('parsed_tradeitem.csv',sep=';',quotechar='"', skipinitialspace=True)
kusok=df.head(100000)
kusok['Tradeitemcountryoforigin']=kusok['Tradeitemcountryoforigin'].apply(countryToInt)
kusok['Tradeitemcountryofassembly']=kusok['Tradeitemcountryofassembly'].apply(countryToInt)
kusok['Ticountryoflastprocessing']=kusok['Ticountryoflastprocessing'].apply(countryToInt)
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
train['OkrbLabel']=okrbLabels.apply(getChapter)
okrbLabels=np.array(okrbLabels)
unnecColumns=['WeightUOM']
train=train.drop(columns=list(nans.loc[:, nans.loc['Freq']>threshold].columns)+classColumns+unnecColumns)  
train=train.sample(n=50000, random_state=42)
okrbLabels=np.array(train['OkrbLabel'])
train=train.drop(columns=['Istradeitemmarkedasrecyclable','PackageLevelCode'])
counts=pd.DataFrame(train.nunique())
func=np.array(train['Functionalname'])