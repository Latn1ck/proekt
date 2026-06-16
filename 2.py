import fasttext
from datetime import datetime


print(f'start {datetime.now()}')
model = fasttext.train_supervised(
    input='train.txt',
    lr=0.1,             
    epoch=25,            
    minn=3,
    maxn=5,        
    dim=512,             
    loss='hs',      
    thread=4,            
    verbose=2    
)
model.save_model("model.bin")
print(f'finish {datetime.now()}')
labels,probas=model.predict('пиджак мужской')
print(labels)