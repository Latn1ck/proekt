from lxml import etree


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
def getIerarchyGPC(brick, tree):
    res=[]
    curr=brick.getparent()
    for i in range(3):
        res.append(curr)
        curr=curr.getparent()
    return res
def getIerarchyOKRB(elem,tree):
    res=[]
    return res
treeOKRB = etree.parse('OKRB007.xml')
rootOKRB = treeOKRB.getroot()
rows=rootOKRB.findall('row')
dictOKRB={}
for row in rows:
    code=getCodeOKRB(row)
    klass=getNameOKRB(row)
    expl=getExplOKRB(row)
    if code is not None and klass is not None:
        if expl is not None:
            dictOKRB[code]=[klass,expl]
        else:
            dictOKRB[code]=[klass,'']

treeGPC = etree.parse('GPC as of November 2021 (GDSN) v20211209 RU.xml')
element = treeGPC.xpath("//brick[@code='10001441']")[0]
rootGPC = treeGPC.getroot()
bricks=rootGPC.findall('.//brick')
dictBrick={}
for i in bricks:
    dictBrick[i.attrib['code']]=[i.attrib['text'],i.attrib['definition'], getAttributes(i)]
print(list(map(getCodeGPC,getIerarchyGPC(bricks[52],treeGPC))))
