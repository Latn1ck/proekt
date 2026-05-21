from datetime import datetime
import DatasetCreation as ch2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tnved 
import pandas as pd

print(f'Начало: {datetime.now()}')
df=ch2.dataset
tnvedData=tnved.df
X=list(df['X'])
y=list(df['y'])
classesCodes=list(tnvedData['Код6'])
classesStrings=list(tnvedData['Наименование'])
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_embeddings=np.load('X_embeddings.npy')
classEmbeddings=np.load('z_embeddings.npy')
classDict={k:v for k,v in zip(y,classEmbeddings)}

#тестовые данные
test=ch2.test
X_test=list(test['X'])
X_test_embeddings=np.load('X_test_embeddings.npy')
yTest=list(test['y'])

#классификация
quantizer = faiss.IndexFlatL2(384)
nlist=4000
nProbe=64
dim=384
k=3
index = faiss.IndexIVFFlat(quantizer,dim,nlist,faiss.METRIC_L2)
index.train(X_embeddings)
index.add(X_test_embeddings)
index.nprobe=nProbe
D, I = index.search(X_test_embeddings, k)
print(I.shape)
yPred1=[y[i] for i in I[:,0]]
yPred2=[y[i] for i in I[:,1]]
yPred3=[y[i] for i in I[:,2]]
final=pd.DataFrame({'X':X_test,'y':yTest,'yPred1':yPred1,'yPred2':yPred2,'yPred3':yPred3})
final.to_excel('final.xlsx')
accuracy=(np.sum(yPred1==yTest)+np.sum(yPred2==yTest)+np.sum(yPred3==yTest))/len(yTest)
print(f'accuracy {accuracy}')
print(f'Конец: {datetime.now()}')