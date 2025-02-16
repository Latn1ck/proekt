import pandas as pd
import xml.etree.ElementTree as ET


chunks=pd.read_csv('C:/проект/parsed_tradeitem.csv',sep=';',chunksize=10000)

for chunk in chunks:
    df=chunk.head(30)
    break
print(df['Functionalname'])

""" treeOKRB = ET.parse('OKRB007.xml')
rootOKRB = treeOKRB.getroot()
iteratorOKRB=rootOKRB.iter()
current=next(iteratorOKRB)
while next(iteratorOKRB) is not None:
    if current.tag=='OKRB_CODE':
        res=f'код: {current.text}'
        current=next(iteratorOKRB)
        res=f'класс: {current.text}, '+res
        print(res)
    current=next(iteratorOKRB) """