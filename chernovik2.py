import pandas as pd
import re
import numpy as np


def getChapter(x):
    if pd.isna(x):
        return 0
    x=re.sub(r'[а-яА-я\s]','',x)
    if len(x)==1:
        return 0
    if x[1] in ['.',',']:
        return int(x[0])
    return int(x[:2])

df=pd.read_csv('parsed_tradeitem.csv',sep=';',quotechar='"',skipinitialspace=True)
df=df.head(500000)
df['OkrbChapter']=df['Okrb007'].apply(getChapter)
dataset=pd.DataFrame({'sample':df['Functionalname'],'label':df['OkrbChapter']})
ix=np.unique(np.array(df['OkrbChapter']))
split=[dataset[dataset['label']==i] for i in ix]
short=[i for i in split if len(i)<1007]
long=[i for i in split if len(i)>1007]
dataset = pd.concat(short, axis=0, ignore_index=True)
for i in long:
    N=int(np.round(np.random.normal(1000, 50)))
    sample=i.sample(n=N,random_state=42)
    dataset=pd.concat([dataset,sample],axis=0,ignore_index=True)
dataset=dataset.sample(frac=1).reset_index(drop=True)