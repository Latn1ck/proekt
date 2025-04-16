import numpy as np
import XMLPreproccesing as ch
import pandas as pd
import DatasetCreation as ch2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.preprocessing import StandardScaler

""" пытаемся классифицировать товары на первом уровне (товару поставить в соответствие раздел). Идея: эмбеддинг+метод классификации из классического машинного обучения """
def plot_hystogram(a, name): #гистограмма частот для выборки разделов
    a_keys, a_counts=np.unique(a, return_counts=True)
    a_values=np.array([i/len(a) for i in a_counts])
    plt.bar(a_keys, a_values)
    plt.xlabel('Классы')
    plt.ylabel('Частоты')
    plt.title(f'Гистограмма {name}')
    plt.show()
def chapterPreprocess(x): #убираем лишнее
    if x[:6]=="РАЗДЕЛ":
        x=x[10:]
    x=x.replace('\n',' ')
    x=x.replace('\t','')
    return x.lower()
def accuracy(pred,ground_truth): #точность классификации
    return (np.sum(pred==ground_truth))/(len(pred))
#классы товаров
""" dictOKRBexpl=ch.dictOKRBexpl
dictOKRBklass=ch.dictOKRBklass
shortDictOKRB={}
for i in dictOKRBexpl.keys():
    if not dictOKRBexpl[i]=='':
        shortDictOKRB[i]=chapterPreprocess(dictOKRBexpl[i])
    else:
        shortDictOKRB[i]=chapterPreprocess(dictOKRBklass[i])
shortDictOKRB={k:chapterPreprocess(v) for k,v in zip(shortDictOKRB.keys(),shortDictOKRB.values()) if len(k)==2} """
#подгрузка датасета
dataset=ch2.dataset
X_train, X_test, y_train, y_test = train_test_split(np.array(dataset['sample']), np.array(dataset['label']), test_size=0.2, random_state=42)
#инициализация модели
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#эмбеддинги описаний классов
trainEmbeddings=model.encode(X_train)
testEmbeddings=model.encode(X_test)
#стандартизация
scaler = StandardScaler()
train_scaled = scaler.fit_transform(trainEmbeddings)
test_scaled = scaler.transform(testEmbeddings)
#логистическая регрессия
LogRegr=LogisticRegression()
LogRegr.fit(train_scaled,y_train)
#предикт
y_predLogRegr=LogRegr.predict(test_scaled)
# plot_hystogram(y_test, 'y_test')
# plot_hystogram(y_predLogRegr, 'Log. Regr.')
#svm
param_grid = {'C': [0.1, 1, 10, 100],'kernel': ['rbf', 'poly'],'gamma': ['scale', 'auto', 0.1, 1]}
svm= SVC(decision_function_shape='ovr',C=10,kernel='rbf',gamma='scale',random_state=42)
svm.fit(train_scaled,y_train)
y_predSVM=svm.predict(test_scaled)
#plot_hystogram(y_predSVM, 'SVM')

#bayes
gnb = GaussianNB()
gnb.fit(train_scaled, y_train)
y_predNB=gnb.predict(test_scaled)
#plot_hystogram(y_predNB, 'Bayes')
#knn
best_K=1
best_acc=0
print(f'Начало: {datetime.now()}')
for k in range(1,30):
    knn=KNeighborsClassifier(n_neighbors=k, metric='cosine',weights='distance')
    knn.fit(train_scaled,y_train)
    y_predKNN=knn.predict(testEmbeddings)
    if accuracy(y_test,y_predKNN)>best_acc:
        best_K=k
        best_acc=accuracy(y_test,y_predKNN)
best_KNN=KNeighborsClassifier(n_neighbors=best_K, metric='cosine',weights='distance')
best_KNN.fit(train_scaled,y_train)
y_predKNN=best_KNN.predict(test_scaled)
print(f'Конец: {datetime.now()}')
accDict={'LogRegr':accuracy(y_test,y_predLogRegr),'SVM':accuracy(y_test,y_predSVM),'bayes':accuracy(y_test,y_predNB),f'KNN, k={best_K}':accuracy(y_test,y_predKNN)}
print(accDict)
best = max(accDict, key=accDict.get)
print(best, f'accuracy={accDict[best]}')
#plot_hystogram(y_predKNN, 'KNN')
res=pd.DataFrame({'sample':X_test, 'true label':y_test,'predictLogRegr':y_predLogRegr, 'predictSVM':y_predSVM, 'predictBayes':y_predNB,'predictKNN':y_predKNN})
res.to_excel('res.xlsx')