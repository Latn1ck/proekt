from lxml import etree
import pandas as pd


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
    res=[]
    curr=elem
    next=getParentOKRB(curr,tree)
    root=tree.getroot()
    while not curr==next:
        curr=getParentOKRB(curr,tree)
        next=getParentOKRB(next,tree)
        res.append(curr)
    return res

treeOKRB = etree.parse('OKRB007.xml')
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
bricks=rootGPC.findall('.//brick')
dictBrickText={}
dictBrickDefinition={}
for i in bricks:
    dictBrickText[i.attrib['code']]=i.attrib['text']
    dictBrickDefinition[i.attrib['code']]=i.attrib['definition']
dictBrickDefinition['0']=''
dictBrickText['0']=''
df=pd.read_csv('C:/проект/parsed_tradeitem.csv',sep=";",low_memory=False)
df=df.head(300)
kusokOKRB=pd.DataFrame(df[['Okrb007', 'GpcBrick', 'Functionalname']])
kusokOKRB['OKRB_class'] = kusokOKRB['Okrb007'].map(dictOKRBklass)
kusokOKRB['OKRB_expl']=kusokOKRB['Okrb007'].map(dictOKRBexpl)
kusokOKRB['GpcBrick'] =kusokOKRB['GpcBrick'].fillna(0).map(int).map(str)
kusokOKRB['GPC_class'] = kusokOKRB['GpcBrick'].map(dictBrickText)
kusokOKRB['GPC_expl']=kusokOKRB['GpcBrick'].map(dictBrickDefinition)
kusokOKRB=kusokOKRB.reindex(columns=['Okrb007','OKRB_class','OKRB_expl','GpcBrick','GPC_class','GPC_expl','Functionalname'])