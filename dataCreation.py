import pandas as pd
from lemmat import lemmatize_sentence
from tqdm import tqdm


tqdm.pandas()
tnved=pd.read_csv('tnved.csv')
df_all=pd.read_parquet('df_all.parquet')
d={k:v for k,v in zip(tnved['Код'].astype(str),tnved['Наименование'])}
def getDescription(x):
    return d[x] if x in d.keys() else ''
desc=[getDescription(i) for i in list(df_all['Tnvedcode'].astype(str))]
data=pd.DataFrame({'Functionalname':df_all['Functionalname'],'Tnvedcode':df_all['Tnvedcode'],'Description':desc})
synth=pd.read_csv('synthData.csv')
synth['Functionalname']=synth['Functionalname'].progress_apply(lemmatize_sentence)
synth['Description']=synth['Tnvedcode'].astype(str).progress_apply(lambda x:getDescription(x+'0'))
data=pd.concat([data,synth],ignore_index=True)
data=data[data['Description']!='']
data.to_csv('data.csv')