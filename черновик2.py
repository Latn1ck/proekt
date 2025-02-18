import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np


df=pd.read_csv('D:/проект/kusok.csv')
text_data=df[df.select_dtypes('object').columns].drop('Okrb007',axis=1)
numerical_data=df[df.select_dtypes(('int64','float64','bool')).columns]
tfidf = TfidfVectorizer()
text_features = np.concatenate((tfidf.fit_transform(text_data).toarray().reshape(50,50),np.empty((250,50))),axis=0)
print(text_features.shape)
combined_data = np.hstack((numerical_data, text_features))
pca = PCA(n_components=50)
reduced_data = pca.fit_transform(combined_data)
print(reduced_data)