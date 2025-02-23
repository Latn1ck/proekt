from lxml import etree
import numpy as np


def getId(x):
    return None if x.find('OKRB_ID') is None else x.find('OKRB_ID').text

def getParentId(x):
    return None if x.find('PARENT_ID') is None else x.find('PARENT_ID').text

def getExpl(x):
    return None if x.find('EXPLANATIONS') is None else x.find('EXPLANATIONS').text

def getName(x):
    return None if x.find('OKRB_NAME') is None else x.find('OKRB_NAME').text

def getCode(x):
    return None if x.find('OKRB_CODE') is None else x.find('OKRB_CODE').text


treeOKRB = etree.parse('OKRB007.xml')
rootOKRB = treeOKRB.getroot()
rows=rootOKRB.findall('row')
dictOKRB={}
for row in rows:
    code=getCode(row)
    klass=getName(row)
    expl=getExpl(row)
    if code is not None and klass is not None:
        if expl is not None:
            dictOKRB[code]=(klass,expl)
        else:
            dictOKRB[code]=(klass,'')
treeGPC = etree.parse('GPC as of November 2021 (GDSN) v20211209 RU.xml')
rootGPC = treeGPC.getroot()
dictBrick={}
for i in rootGPC.findall('.//brick'):
    dictBrick[i.attrib['text']]=(i.attrib['code'],i.attrib['definition'])
dictClass={}
iterator=rootGPC.iter()

for i in iterator:
    if i.tag in ['family','class','segment']:
        dictClass[i.attrib['code']]=(i.attrib['text'],i.attrib['definition'])
""" for i in rootGPC.findall('.//family','.//class','.//segment'):
    dictClass[i.attrib['text']]=(i.attrib['code'], i.attrib['definition'])
print(dictClass) """
print(dictClass)