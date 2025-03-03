from lxml import etree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def getIdOKRB(x):
    return None if x.find('OKRB_ID') is None else x.find('OKRB_ID').text
def getParentIdOKRB(x):
    return None if x.find('PARENT_ID') is None else x.find('PARENT_ID').text
def getExplOKRB(x):
    return None if x.find('EXPLANATIONS') is None else x.find('EXPLANATIONS').text
def getNameOKRB(x):
    return None if x.find('OKRB_NAME') is None else x.find('OKRB_NAME').text
def getCodeOKRB(x):
    return None if x.find('OKRB_CODE') is None else x.find('OKRB_CODE').text
def getAttributes(brick):
    return [i.attrib for i in brick.findall('attType')]
def getCodeGPC(x):
    return x.attrib['code']
def getTextGPC(x):
    return x.attrib['text']
def getDefinitionGPC(x):
    return x.attrib['definition']
def getIerarchyGPC(brick,root):
    if brick==root:
        return 0
    res=[]
    curr=brick.getparent()
    while not curr==root:
        res.append(curr)
        curr=curr.getparent()
    return res
def getParentOKRB(elem,tree):
    elemParentID=getParentIdOKRB(elem)
    t=tree.xpath(f"//row[OKRB_ID[contains(text(), '{elemParentID}')]]")
    return None if t==[] else t[0]
def getIerarchyOKRB(elem,tree):
    if elem==tree.getroot():
        return 0
    res=[]
    curr=elem
    next=getParentOKRB(curr,tree)
    while not curr==next:
        curr=getParentOKRB(curr,tree)
        next=getParentOKRB(next,tree)
        res.append(curr)
    return res
def getCodeTNVED(x):
    return x.attrib['code'] if 'code' in x.attrib.keys() else None
def getNameTNVED(x):
    return x.attrib['name'] if 'name' in x.attrib.keys() else ''
def getIerarchyTNVED(item,root):
    if item==root:
        return 0
    res=[]
    curr=item.getparent()
    while not curr==root:
        res.append(curr)
        curr=curr.getparent()
    return res
def transform(brick):
    return np.nan if np.isnan(brick) else int(brick)

treeOKRB = etree.parse('OKRB007.xml') #depth=8
rootOKRB = treeOKRB.getroot()
rows=rootOKRB.findall('row')
dictOKRBklass={}
dictOKRBexpl={}
for row in rows:
    code=getCodeOKRB(row)
    klass=getNameOKRB(row)
    expl=getExplOKRB(row)
    if code is not None and klass is not None:
        if expl is not None:
            dictOKRBklass[code]=klass
            dictOKRBexpl[code]=expl
        else:
            dictOKRBklass[code]=klass
            dictOKRBexpl[code]=''
treeGPC = etree.parse('GPC as of November 2021 (GDSN) v20211209 RU.xml')
rootGPC = treeGPC.getroot()
bricks=rootGPC.findall('.//brick') #все брики
dictBrickText={}
dictBrickDefinition={}
for i in bricks:
    dictBrickText[i.attrib['code']]=i.attrib['text']
    dictBrickDefinition[i.attrib['code']]=i.attrib['definition']
dictBrickDefinition['0']=''
dictBrickText['0']=''
maxDepthGPC=max([len(getIerarchyGPC(i,rootGPC)) for i in bricks]) #3
depths=[len(getIerarchyGPC(i,rootGPC)) for i in bricks]

treeTNVED=etree.parse('tnved.xml')
rootTNVED=treeTNVED.getroot()
items=rootTNVED.findall('.//item')
maxDepthTNVED=max([len(getIerarchyTNVED(i,rootTNVED)) for i in items]) #12
parentsTNVED=list(set([i.getparent() for i in rootTNVED.iter()]))
leavesTNVED=[i for i in rootTNVED.iter() if not i in parentsTNVED]
depths=[len(getIerarchyTNVED(i,rootTNVED)) for i in leavesTNVED]
dictItemName={}
for i in rootTNVED.iter():
    code=getCodeTNVED(i)
    name=getNameTNVED(i)
    if not code is None:
        dictItemName[code]=name

df=pd.read_excel('D:/проект/kusok.xlsx')
col=list(df['GpcBrick'])
keys=list(set(col))
values=[col.count(i) for i in keys]
plt.bar(keys, values, edgecolor='black')
plt.title('Гистограмма частот')
plt.xlabel('Брики')
plt.ylabel('Частоты')
plt.show()