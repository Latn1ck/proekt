import pymorphy3
from nltk.tokenize import word_tokenize
import re


morph = pymorphy3.MorphAnalyzer()
def lemmatize_sentence(text):
    text_cleaned = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text_cleaned, language='russian')
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)