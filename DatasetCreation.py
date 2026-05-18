import pandas as pd
import numpy as np
import XMLPreproccesing as ch1

df=pd.read_csv('df_all.txt')
X=df['Functionalname']+' '+df['Variant']+' '+df['Consist']
y=df['Tnvedcode'].apply(lambda x:x[:4]+x[5:7])
d=ch1.dictSix
z=y.apply(lambda x:d[x])
dataset=pd.DataFrame({'X':X,'y':y,'z':z})
dataset.sample(1000).to_excel('head1.xlsx')