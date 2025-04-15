import pandas as pd
import re
import numpy as np


def getChapter(x): #функция принимает код и возвращает раздел: '10.13.12.100' -> 10 
    if pd.isna(x):
        return 0
    x=re.sub(r'[а-яА-я\s]','',x)
    if len(x)==1:
        return 0
    if x[1] in ['.',',']:
        return int(x[0])
    return int(x[:2])

df=pd.read_csv('parsed_tradeitem.csv',sep=';',quotechar='"',skipinitialspace=True,chunksize=500000) #открываем датасет партиями по 500000 записей
df=next(df)
df['OkrbChapter']=df['Okrb007'].apply(getChapter) #новая колонка, содержащая только раздел
dataset=pd.DataFrame({'sample':df['Functionalname'],'label':df['OkrbChapter']})
ix=np.unique(np.array(df['OkrbChapter']))
split=[dataset[dataset['label']==i] for i in ix] #делим датасет на части по номеру раздела
short=[i for i in split if len(i)<1007]
long=[i for i in split if len(i)>1007]
#составляем датасет: добавляем все записи, главы которых встречаются редко (<1007 раз) и добавляем некоторое число записей (~Norm(1000,50)) из частей, главы которых встречаются часто
dataset = pd.concat(short, axis=0, ignore_index=True)
for i in long:
    N=int(np.round(np.random.normal(1000, 50)))
    sample=i.sample(n=N,random_state=42)
    dataset=pd.concat([dataset,sample],axis=0,ignore_index=True)
dataset=dataset.sample(frac=1).reset_index(drop=True)