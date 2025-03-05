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

df=ch.df
nans=pd.DataFrame(df.isna().sum())
nans=nans.rename(columns={0:'score'})
nans['Freq']=nans['score']/df.shape[0]
nans.to_excel('nans.xlsx')