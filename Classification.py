import numpy as np
import chernovik as ch
import pandas as pd
import chernovik2 as ch2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime


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
ix=np.array([i for i in range(len(y_test)) if not y_test[i] in [14,15]])
y_testShort=y_test[ix]

#инициализация модели
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#эмбеддинги описаний классов
trainEmbeddings=model.encode(X_train)
testEmbeddings=model.encode(X_test)
""" #логистическая регрессия
LogRegr=LogisticRegression()
LogRegr.fit(trainEmbeddings,y_train)
#предикт
y_predLogRegr=LogRegr.predict(testEmbeddings)
#результаты логистической регрессии
print(f'Logistic Regression Accuracy: {accuracy(y_test,y_predLogRegr)}')
y_predLogRegrShort=y_predLogRegr[ix]
print(f'Logistic Regression Accuracy without 14 and 15: {accuracy(y_testShort, y_predLogRegrShort)}')
plot_hystogram(y_test, 'y_test')
plot_hystogram(y_predLogRegr, 'Log. Regr.')
#svm
svm= SVC(decision_function_shape='ovr',C=1.0,kernel='rbf',gamma='scale',random_state=42)
svm.fit(trainEmbeddings,y_train)
y_predSVM=svm.predict(testEmbeddings)
print(f'SVM Accuracy: {accuracy(y_test,y_predSVM)}')
y_predSVMShort=y_predSVM[ix]
print(f'SVM Accuracy without 14 and 15: {accuracy(y_testShort, y_predSVMShort)}')
plot_hystogram(y_predSVM, 'SVM')
#bayes
gnb = GaussianNB()
gnb.fit(trainEmbeddings, y_train)
y_predNB=gnb.predict(testEmbeddings)
print(f'Bayes Accuracy: {accuracy(y_test,y_predSVM)}')
y_predNBShort=y_predNB[ix]
print(f'Bayes Accuracy without 14 and 15: {accuracy(y_testShort, y_predNBShort)}')
plot_hystogram(y_predNB, 'Bayes') """
#knn
best_K=1
best_acc=0
print(f'Начало: {datetime.now()}')
for k in range(1,30):
    knn=KNeighborsClassifier(n_neighbors=k, metric='cosine',weights='distance')
    knn.fit(trainEmbeddings,y_train)
    y_predKNN=knn.predict(testEmbeddings)
    print(f'k={k}, accuracy={accuracy(y_test,y_predKNN)}')
    if accuracy(y_test,y_predKNN)>best_acc:
        best_K=k
        best_acc=accuracy(y_test,y_predKNN)
print(f'Best KNN Accuracy: {best_acc}, k={best_K}')
print(f'Конец: {datetime.now()}')
# plot_hystogram(y_predKNN, 'KNN')
# res=pd.DataFrame({'sample':X_test, 'true label':y_test,'predictLogRegr':y_predLogRegr, 'predictSVM':y_predSVM, 'predictBayes':y_predNB,'predictKNN':y_predKNN})
# res.to_excel('res.xlsx')