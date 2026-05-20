from datetime import datetime
import DatasetCreation as ch2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tnved 

print(f'Начало: {datetime.now()}')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
df=ch2.dataset
tnvedData=tnved.df
X=list(df['X'])
y=list(df['y'])
classesCodes=list(tnved.df['Код6'])
classesStrings=list(tnved.df['Наименование'])
X_embeddings=np.load('X_embeddings.npy')
classEmbeddings=np.load('z_embeddings.npy')
classDict={k:v for k,v in zip(y,classEmbeddings)}
index=faiss.IndexFlatIP(classEmbeddings.shape[1])
index.add(classEmbeddings.astype(np.float32))

print(f'Конец: {datetime.now()}')