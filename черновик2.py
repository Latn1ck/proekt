import pandas as pd
import xml.etree.ElementTree as ET


treeGPC=ET.parse('GPC as of November 2021 (GDSN) v20211209 RU.xml')
rootGPC=treeGPC.getroot()
iteratorGPC=rootGPC.iter()
current=next(iteratorGPC)
current=next(iteratorGPC)
while next(iteratorGPC) is not None:
    if current.tag=='class':
        print(current.attrib['text'])
    current=next(iteratorGPC)