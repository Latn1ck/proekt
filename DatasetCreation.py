import pandas as pd
import tnved
import re
import lemmat
from tqdm.auto import tqdm

def clean_class_text(text):
    text = re.sub(r'\(с\s+[\d\.]+\).*?$', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
df=pd.read_csv('df_all.txt')
X= df['Functionalname']
y=list(df['Tnvedcode'].str.replace(' ', '', regex=False).astype(str).str[:6])
l=lambda x:tnved.dict[x] if x in tnved.dict.keys() else ''
z=list(map(l,y))

dataset=pd.DataFrame({'X':X,'y':y,'z':z})
dataset=dataset[dataset['y'].str.len()>4]
dataset=dataset[dataset['z']!='']
dataset=dataset.sample(1000000,random_state=42)
diff=df[~df.index.isin(dataset.index)]
diff=diff.sample(1000000,random_state=42)
diff=diff['Functionalname']
tqdm.pandas()
dataset['z']=dataset['z'].apply(clean_class_text)
dataset['z']=dataset['z'].progress_apply(lemmat.lemmatize_sentence)
dataset=dataset.reset_index(drop=True)
negative=tnved.df.sample(1000000,replace=True)
negative=negative['Наименование']
dataset['negative']=negative.reset_index(drop=True)
dataset['negative']=dataset['negative'].progress_apply(lemmat.lemmatize_sentence)
dataset.to_parquet('dataset.parquet', index=False)
dataset.sample(1000).to_excel('datahead1.xlsx')
test=dataset.sample(50000, random_state=42)