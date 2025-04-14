import numpy as np
import chernovik as ch
import pandas as pd
import chernovik2 as ch2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def plot_hystogram(a, name):
    a_keys, a_counts=np.unique(a, return_counts=True)
    a_values=np.array([i/len(a) for i in a_counts])
    plt.bar(a_keys, a_values)
    plt.xlabel('Классы')
    plt.ylabel('Частоты')
    plt.title(f'Гистограмма {name}')
    plt.show()

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
#подгрузка датасета
dataset=ch2.dataset
X_train, X_test, y_train, y_test = train_test_split(np.array(dataset['sample']), np.array(dataset['label']), test_size=0.2, random_state=42)
#инициализация модели
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#эмбеддинги описаний классов
print('начинаем делать эмбеддинги')
trainEmbeddings=model.encode(X_train)
testEmbeddings=model.encode(X_test)
print('эмбеддинги готовы')
#логистическая регрессия
LogRegr=LogisticRegression()
LogRegr.fit(trainEmbeddings,y_train)
#предикт
y_predLogRegr=LogRegr.predict(testEmbeddings)
#результаты логистической регрессии
""" считаем точность, делаем датафрейм с тестовыми образцами и предиктами,рисуем гистограммы, составляем confusion matrix """
print(f'Logistic Regression Accuracy: {accuracy(y_test,y_predLogRegr)}')
ix=np.array([i for i in range(len(y_test)) if not y_test[i] in [14,15]])
y_testShort=y_test[ix]
y_predShort=y_predLogRegr[ix]
print(f'Logistic Regression Accuracy without 14 and 15: {accuracy(y_testShort, y_predShort)}')
ix=np.array(['00']+list(shortDictOKRB.keys())).astype(int)
N=len(ix)

matrixLogRegr=np.zeros((N,N),dtype=int)
for i in range(N):
    for j in range(N):
        matrixLogRegr[i][j]=len([k for k in range(len(y_test)) if y_test[k]==ix[i] and y_predLogRegr[k]==ix[j]])
matLogRegr=pd.DataFrame(matrixLogRegr,columns=[f'predicted_{i}' for i in ix], index=[f'actual_{i}' for i in ix])
matLogRegr.to_excel('ConfusionMatrixLogRegr.xlsx')
plot_hystogram(y_test, 'y_test')
plot_hystogram(y_predLogRegr, 'y_pred')
#svm
svm= SVC(decision_function_shape='ovr',C=1.0,kernel='rbf',gamma='scale',random_state=42)
svm.fit(trainEmbeddings,y_train)
y_predSVM=svm.predict(testEmbeddings)
print(f'SVM Accuracy: {accuracy(y_test,y_predSVM)}')
ix=np.array([i for i in range(len(y_test)) if not y_test[i] in [14,15]])
y_predSVMShort=y_predSVM[ix]
print(f'SVM Accuracy without 14 and 15: {accuracy(y_testShort, y_predSVMShort)}')
matrixSVM=np.zeros((N,N),dtype=int)
for i in range(N):
    for j in range(N):
        matrixSVM[i][j]=len([k for k in range(len(y_test)) if y_test[k]==ix[i] and y_predSVM[k]==ix[j]])
matSVM=pd.DataFrame(matrixSVM,columns=[f'predicted_{i}' for i in ix], index=[f'actual_{i}' for i in ix])
matSVM.to_excel('ConfusionMatrixSVM.xlsx')
plot_hystogram(y_predSVM, 'y_SVM')
#bayes


res=pd.DataFrame({'sample':X_test, 'true label':y_test,'predictLogRegr':y_predLogRegr, 'predictSVM':y_predSVM})
res.to_excel('res.xlsx')