import pandas as pd
import tnved
df=pd.read_csv('df_all.txt')
X= list(df['Functionalname'].fillna('') + ' ' +df['Variant'].fillna('') + ' ' +df['Consist'].fillna(''))
y=list(df['Tnvedcode'].str.replace(' ', '', regex=False).astype(str).str[:6])
l=lambda x:tnved.dict[x] if x in tnved.dict.keys() else ''
z=list(map(l,y))
dataset=pd.DataFrame({'X':X,'y':y,'z':z})
dataset=dataset[dataset['y'].str.len()>4]
dataset=dataset[dataset['z']!='']
dataset.sample(1000).to_excel('datahead1.xlsx')