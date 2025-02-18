import pandas as pd
from lxml import etree
import numpy as np

treeOKRB = etree.parse('OKRB007.xml')
rootOKRB = treeOKRB.getroot()
print(help(rootOKRB))
keys=[]
values=[]
asd=[]
notFound=0
for row in rootOKRB.findall('row'):
    code=row.find('OKRB_CODE')
    klass=row.find('OKRB_NAME')
    expl=row.find('EXPLANATIONS')
    if code is not None and klass is not None:
        keys.append(klass.text)
        if expl is not None:
            values.append((code.text,expl.text))
        else:
            values.append((code.text,''))
#dictOKRB={k:v for k,v in zip(keys,values)}

treeGPC = etree.parse('GPC as of November 2021 (GDSN) v20211209 RU.xml')
rootGPC = treeGPC.getroot()
iteratorGPC=rootGPC.iter()
keys=[]
values=[]
for i in rootGPC.findall('.//brick'):
    keys.append(i.attrib['text'])
    values.append(i.attrib['code'])
dictBrick={k:v for k,v in zip(keys,values)}
#print(dictBrick)
keys=[]
values=[]
for i in rootGPC.findall('.//class'):
    keys.append(i.attrib['text'])
    values.append(i.attrib['code'])
dictClass={k:v for k,v in zip(keys,values)}
#print(dictClass)
"""
while current is not None:
    if current.tag=='OKRB_CODE':
        keys.append(current.text)
        current=next(iteratorGPC)
        values.append(current.text)
    current=next(iteratorGPC)
 """