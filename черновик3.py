from sentence_transformers import SentenceTransformer, util
import numpy as np
import chernovik as ch
import pandas as pd
import chernovik2 as ch2
import random


def chapterPreprocess(x):
    x=x[10:]
    x=x.replace('\n',' ')
    x=x.replace('\t','')
    return x.lower()
def accuracy(pred,ground_truth):
    return (np.sum(pred==ground_truth))/(len(pred))
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') #SBERT
dictOKRB=ch.dictOKRBklass
shortDictOKRB={k:chapterPreprocess(v) for k,v in zip(dictOKRB.keys(),dictOKRB.values()) if len(k)==2}
X=ch2.func
yTrue=ch2.okrbLabels
ix=np.array(random.sample(list(range(len(X))),k=1000))
XSample=X[ix]
ySample=yTrue[ix]
reference_texts={k:v for k,v in zip(shortDictOKRB.keys(),shortDictOKRB.values())}
ref_embeddings = {label: model.encode(texts) for label, texts in reference_texts.items()}
yPred=np.zeros(ySample.shape)
for i in range(len(XSample)):
    embedding = model.encode(XSample[i])
    scores = {label: util.cos_sim(embedding, embeddings).mean() for label, embeddings in ref_embeddings.items()}
    yPred[i] = max(scores, key=scores.get)
res=pd.DataFrame({'Functional name':XSample, 'True label':ySample,'Prediction':yPred})
res.to_excel('res.xlsx')
print(f'Accuracy: {accuracy(yPred,ySample)}')
""" # Эталонные предложения и их классы
reference_texts = {
    "одежда": ["ОДЕЖДА"],
    "еда": ["ПРОДУКТЫ ПИЩЕВЫЕ", "НАПИТКИ", "АЛКОГОЛЬНЫЕ НАПИТКИ"]
}

# Эмбеддинги эталонов
ref_embeddings = {label: model.encode(texts) for label, texts in reference_texts.items()}

# Классификация нового предложения
new_text = ["Ветровка","Шоколад","Сидр","Костюм", "Водка", "Чулки"]
for i in new_text:
    new_embedding = model.encode(i)

# Сравнение с эталонами
    scores = {label: util.cos_sim(new_embedding, embeddings).mean() for label, embeddings in ref_embeddings.items()}
    predicted_class = max(scores, key=scores.get)
    print(f"{i}: класс {predicted_class}, схожести: {scores}")
    print() """