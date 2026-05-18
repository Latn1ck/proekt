import numpy as np
import XMLPreproccesing as ch
import pandas as pd
import DatasetCreation as ch2
import matplotlib.pyplot as plt
import faiss
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.preprocessing import StandardScaler

start=datetime.now()
print(f'Начало: {start}')
X,y=ch2.X,ch2.y
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
finish=datetime.now()
print(f'Конец: {finish}')