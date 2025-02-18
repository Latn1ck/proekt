import pandas as pd
from lxml import etree


treeOKRB = etree.parse('OKRB007.xml')
rootOKRB = treeOKRB.getroot()
keys=[]
values=[]
notFound=0
for row in rootOKRB.findall('row'):
    code=row.find('OKRB_CODE')
    klass=row.find('OKRB_NAME')
    if code is not None and klass is not None:
        values.append(code.text)
        keys.append(klass.text)
    else:
        print('Element not found')
        notFound+=1
dictOKRB={k:v for k,v in zip(keys,values)}
print(dictOKRB)
print(notFound)
treeGPC = etree.parse('GPC as of November 2021 (GDSN) v20211209 RU.xml')
rootGPC = treeGPC.getroot()
iteratorGPC=rootGPC.iter()
keys=[]
values=[]
for i in rootGPC.findall('.//brick'):
    keys.append(i.attrib['text'])
    values.append(i.attrib['code'])
dictBrick={k:v for k,v in zip(keys,values)}
print(dictBrick)
keys=[]
values=[]
for i in rootGPC.findall('.//class'):
    keys.append(i.attrib['text'])
    values.append(i.attrib['code'])
dictClass={k:v for k,v in zip(keys,values)}
print(dictClass)
"""
while current is not None:
    if current.tag=='OKRB_CODE':
        keys.append(current.text)
        current=next(iteratorGPC)
        values.append(current.text)
    current=next(iteratorGPC)
 """