import fasttext.util
import numpy as np
import chernovik as ch
import pandas as pd
import chernovik2 as ch2


def cosine_similarity(a,b):
    if np.all(a==0) or np.all(b==0):
        return 0
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
def chapterPreprocess(x):
    if x[:6]=="РАЗДЕЛ":
        x=x[10:]
    x=x.replace('\n',' ')
    x=x.replace('\t','')
    return x.lower()
def accuracy(pred,ground_truth):
    return (np.sum(pred==ground_truth))/(len(pred))
#подгрузка датасета
X_train=ch2.X_train
X_test=ch2.X_test
y_train=ch2.y_train
y_test=ch2.y_test
#классы товаров
dictOKRBexpl=ch.dictOKRBexpl
dictOKRBklass=ch.dictOKRBklass
shortDictOKRB={}
for i in dictOKRBexpl.keys():
    if not dictOKRBexpl[i]=='':
        shortDictOKRB[i]=chapterPreprocess(dictOKRBexpl[i])
    else:
        shortDictOKRB[i]=chapterPreprocess(dictOKRBklass[i])
shortDictOKRB={k:chapterPreprocess(v) for k,v in zip(shortDictOKRB.keys(),shortDictOKRB.values()) if len(k)==2}
shortDictOKRB['00']=''
shortDictOKRB=dict(sorted(shortDictOKRB.items()))
#инициализация модели
fasttext.util.download_model('ru', if_exists='ignore')
# Загрузить модель
model = fasttext.load_model('cc.ru.300.bin')
#эмбеддинги описаний классов
OKRBEmbeddings = {k:model.get_sentence_vector(v) for k,v in zip(shortDictOKRB.keys(),shortDictOKRB.values())}
keys=np.array(list(OKRBEmbeddings.keys()))
#fine tuning

#предикт
y_pred=np.zeros(y_test.shape)
for i in range(len(y_test)):
    newEmb=model.get_sentence_vector(X_test[i])
    sims=np.array([cosine_similarity(newEmb,j) for j in OKRBEmbeddings.values()])
    result=keys[np.argmax(sims)]
    y_pred[i]=result
#результаты
""" считаем точность, делаем датафрейм с тестовыми образцами и предиктами """
print(f'Accuracy: {accuracy(y_test,y_pred)}')
res=pd.DataFrame({'sample':X_test, 'true label':y_test,'predict':y_pred})
res.to_excel('res.xlsx')