from datetime import datetime
import DatasetCreation as ch2
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer,SentenceTransformerTrainer,SentenceTransformerTrainingArguments
from sentence_transformers.losses import GISTEmbedLoss
import faiss
import numpy as np
import tnved 
import pandas as pd
from datasets import Dataset

print(f'Начало: {datetime.now()}')
df=ch2.dataset
tnvedData=tnved.df
X=list(df['X'])
y=list(df['y'])
z=list(df['z'])
negative=list(df['negative'])
classesCodes=list(tnvedData['Код6'])
classesStrings=list(tnvedData['Наименование'])
#fine tuning
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
trainData=[{'query':X[i],'positive':z[i],'negative':negative[i]} for i in range(len(X))]
trainDataset=Dataset.from_list(trainData)
guide_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
loss = GISTEmbedLoss(model, guide_model=guide_model)
args = SentenceTransformerTrainingArguments(
    output_dir="finetuned_product_classifier", # Папка для сохранения модели
    num_train_epochs=3,                         # Количество эпох
    per_device_train_batch_size=32,             # Размер батча (чем больше, тем лучше, но зависит от VRAM)
    learning_rate=2e-5,                         # Скорость обучения (стандартное значение для fine-tuning)
    warmup_ratio=0.1,                           # Доля шагов для прогрева learning rate
    save_strategy="epoch",                      # Сохранять модель после каждой эпохи
    logging_steps=100,                          # Логировать каждые 100 шагов
)
trainer = SentenceTransformerTrainer(model=model,args=args,train_dataset=trainDataset,loss=loss)
trainer.train()
model.save_pretrained("finetuned_product_classifier/final")
#классификация
nlist=4000
nProbe=64
dim=384
k=3
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer,dim,nlist,faiss.METRIC_INNER_PRODUCT)

print(f'Конец: {datetime.now()}')