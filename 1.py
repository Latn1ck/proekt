import pandas as pd
from sklearn.model_selection import train_test_split



data=pd.read_csv('data.csv')
X=list(data['Functionalname'])
y=list(data['Tnvedcode'].astype(str).apply(lambda x:f'__label__{x[:2]} __label__{x[:4]} __label__{x[:6]}'))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05)
with open('train.txt', 'w', encoding='utf-8') as f:
    for i in range(len(X_train)):
        f.write(f'{y_train[i]} {X_train[i]}\n')
test=pd.DataFrame({'X':X_test,'y':y_test})
with open('test.txt', 'w', encoding='utf-8') as f:
    for i in range(len(X_test)):
        f.write(f'{y_test[i]} {X_test[i]}\n')