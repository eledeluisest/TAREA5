#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from importlib import reload
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import *

# Para descarga de las stopwords

reload(sys)
# nltk.download('stopwords')
# sys.setdefaultencoding('utf8')

# Lectura del fichero de texto
f = open('data/Opera - Wikipedia.html', encoding='utf-8')
html = f.read()

def procesa_text(texto_html):
    # Objeto beautifulsoup
    soup = BeautifulSoup(html)
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out
    # get text
    text = soup.get_text().lower()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    # Tokenizamos. Nos quedamos con los símbolos ya que si son igual de frecuentes en todos los textos no aportarán
    # demaisiado en la clasificación pero una diferencia si podría ser significativa entre textos más y menos descriptivos
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    # Evitamos las stop_words del inglés
    filtered_word = [w for w in words if not w in stop_words]
    # Inicializamos el Stemmer de Porter
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in filtered_word]
