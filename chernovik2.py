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
df=pd.read_csv('parsed_tradeitem.csv',sep=';',quotechar='"', skipinitialspace=True)
kusok=df.head(100000)
print(list(set(list(kusok['NetcontentUOM']))))
kusok['Tradeitemcountryoforigin']=kusok['Tradeitemcountryoforigin'].apply(countryToInt)
kusok['Tradeitemcountryofassembly']=kusok['Tradeitemcountryofassembly'].apply(countryToInt)
kusok['Ticountryoflastprocessing']=kusok['Ticountryoflastprocessing'].apply(countryToInt)
nans=pd.DataFrame(kusok.isna().sum())
nans=nans.rename(columns={0:'score'})
nans['Freq']=nans['score']/kusok.shape[0]
nans=nans.T
nans.to_excel('nans.xlsx')
threshold=0.9
train=kusok.copy()
okrbLabels=train['Okrb007']
train['OkrbLabel']=okrbLabels.apply(getChapter)
bricks=train['GpcBrick']
gpcLabels=train[['GpcSegm','GpcFamily','GpcClass']]
tnvedLabels=train[['Tnvedcode','Tnvedpath']]
classColumns=['Okrb007Path','Okrb007','GpcBrick','GpcSegm','GpcFamily','GpcClass','Tnvedcode','Tnvedpath']
unnecColumns=['WeightUOM']
train=train.drop(columns=list(nans.loc[:, nans.loc['Freq']>threshold].columns)+classColumns+unnecColumns)  
train=train.sample(n=50000, random_state=42)
train.to_excel('train.xlsx')
model = nn.Sequential(
    nn.Linear(10, 20),  # Вход размером 10, выход 20
    nn.ReLU(),          # Функция активации
    nn.Linear(20, 1)    # Вход размером 20, выход 1
)

print(model)