import pandas as pd
import tnved
import re


def clean_product_text(text):
    text = re.sub(r'артикул\s+\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'модель\s+\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{2,3}-\d{2,3}-\d{2,3}', '', text)
    text = re.sub(r'сорт\s+\d+', '', text, flags=re.IGNORECASE)
    text = ' '.join(text.split())
    return text
def clean_class_text(text):
    text = re.sub(r'\(с\s+[\d\.]+\).*?$', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
df=pd.read_csv('df_all.txt')
X= df['Functionalname'].fillna('') + ' ' +df['Variant'].fillna('') + ' ' +df['Consist'].fillna('')
y=list(df['Tnvedcode'].str.replace(' ', '', regex=False).astype(str).str[:6])
l=lambda x:tnved.dict[x] if x in tnved.dict.keys() else ''
z=list(map(l,y))
dataset=pd.DataFrame({'X':X,'y':y,'z':z})
dataset=dataset[dataset['y'].str.len()>4]
dataset=dataset[dataset['z']!='']
dataset=dataset.sample(1000000)
dataset['X']=dataset['X'].apply(clean_product_text)
dataset['z']=dataset['z'].apply(clean_class_text)
dataset.sample(1000).to_excel('datahead1.xlsx')