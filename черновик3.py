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
shortDictOKRB[0]=''
shortDictOKRB=dict(sorted(shortDictOKRB.items()))
shortDictOKRBReversed={v:k for k,v in zip(shortDictOKRB.keys(),shortDictOKRB.values())}
#подгрузка датасета
train_df=ch2.train_df
print(train_df)
print(train_df['label'].dtype)
train_df['fasttext']='__label__' + train_df['label'].map(shortDictOKRBReversed) + ' ' + train_df['sample']
print(train_df['fasttext'])
train_df['fasttext'].to_csv("train.txt", index=False, header=False, sep="\n")
test_df=ch2.test_df
X_test=np.array(test_df['sample'])
y_test=np.array(test_df['label'])
#инициализация модели
fasttext.util.download_model('ru', if_exists='ignore')
# Загрузить модель
model = fasttext.load_model('cc.ru.300.bin')
#эмбеддинги описаний классов
OKRBEmbeddings = {k:model.get_sentence_vector(v) for k,v in zip(shortDictOKRB.keys(),shortDictOKRB.values())}
keys=np.array(list(OKRBEmbeddings.keys()))
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

#fine tuning
надо привести датасет в надлежащий вид
modelFT = fasttext.train_supervised(input='train.txt', dim=300,epoch=50,lr=0.1, wordNgrams=2,verbose=2 )
#новые эмбеддинги описаний классов
OKRBEmbeddingsFT = {k:model.get_sentence_vector(v) for k,v in zip(shortDictOKRB.keys(),shortDictOKRB.values())}
keys=np.array(list(OKRBEmbeddingsFT.keys()))
#новый предикт
y_predFT=np.zeros(y_test.shape)
for i in range(len(y_test)):
    newEmb=modelFT.get_sentence_vector(X_test[i])
    sims=np.array([cosine_similarity(newEmb,j) for j in OKRBEmbeddingsFT.values()])
    result=keys[np.argmax(sims)]
    y_predFT[i]=result
#новые результаты
print(f'new accuracy: {accuracy(y_test,y_predFT)}')
res['predict FT']=y_predFT
res.to_excel('res.xlsx')