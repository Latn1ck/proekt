import fasttext


model=fasttext.load_model('new.bin')
""" lr=0.05,
    epoch=50,
    dim=300,
    minn=2,
    maxn=3,
    loss='hs',
    threads=16,
    verbose=2 """
labels,probas=model.predict('водка')
print(labels,probas)

def getEmb(x,model):
    return model.get_word_vector(x)
print(getEmb('пиджак мужской',model))