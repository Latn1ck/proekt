import pandas as pd
from lxml import etree
import numpy as np

treeOKRB = etree.parse('OKRB007.xml')
rootOKRB = treeOKRB.getroot()
keys=[]
values=[]
notFound=0
rows=rootOKRB.findall('row')
ids=[row.find('OKRB_ID').text for row in rows]
parents=[row.find('PARENT_ID').text for row in rows]
parents=[i for i in parents if parents.count(i)==1]
leaves=[row for row in rows if not (row.find('OKRB_ID').text in parents)]
for i in leaves:
    pass
for row in rows:
    code=row.find('OKRB_CODE')
    klass=row.find('OKRB_NAME')
    expl=row.find('EXPLANATIONS')
    id=row.find('OKRB_ID')
    parentID=row.find('PARENT_ID')
    if code is not None and klass is not None:
        keys.append(code.text)
        if expl is not None:
            values.append((klass.text,expl.text))
        else:
            values.append((klass.text,''))
dictOKRB={k:v for k,v in zip(keys,values)}
treeGPC = etree.parse('GPC as of November 2021 (GDSN) v20211209 RU.xml')
rootGPC = treeGPC.getroot()
iteratorGPC=rootGPC.iter()
keys=[]
values=[]
for i in rootGPC.findall('.//brick'):
    keys.append(i.attrib['text'])
    values.append(i.attrib['code'])
dictBrick={k:v for k,v in zip(keys,values)}

keys=[]
values=[]
for i in rootGPC.findall('.//class'):
    keys.append(i.attrib['text'])
    values.append(i.attrib['code'])
dictClass={k:v for k,v in zip(keys,values)}
"""
while current is not None:
    if current.tag=='OKRB_CODE':
        keys.append(current.text)
        current=next(iteratorGPC)
        values.append(current.text)
    current=next(iteratorGPC)
 """