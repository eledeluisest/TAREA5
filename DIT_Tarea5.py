"""
10/03/2020
Tarea 5 de descubrimiento de información en textos.

Para descarga de las stopwords
nltk.download('stopwords')

"""
import sys
import nltk
import os
from importlib import reload
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re

reload(sys)


# Función que procesa el html devuelve las palabras en una lista después de obtener el contenido,
# aplicar tokenizador, quitar stopwords y aplicar lematizador de Porter
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
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    # Evitamos las stop_words del inglés
    filtered_word = [w for w in words if not w in stop_words]
    # Inicializamos el Stemmer de Porter
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in filtered_word]


t_tot = []
# Lectura del fichero de texto
list_fdf = []
documentos = []
i = 0
for fichero in os.walk('data/'):
    if fichero[0] == 'data/':
        print(fichero)
        print(fichero[2])
        print([fi for fi in fichero[2] if '.html' in fi])
        for fich in [fi for fi in fichero[2] if '.html' in fi]:
            documentos.append([i, fich])
            f = open('data/' + fich, encoding='utf-8')
            html = f.read()
            t1 = procesa_text(html)
            list_fdf.append(dict(nltk.FreqDist(word.lower() for word in t1)))
            t_tot = t1 + t_tot
            i = i + 1
vocabulario = set(t_tot)
indice = pd.DataFrame(tuple(vocabulario), columns=['vocabulario']).sort_values('vocabulario').reset_index(
    drop=True).set_index('vocabulario')
indice['index'] = np.arange(len(indice))
frecuencias = indice.copy()
for indice_doc, nombre in documentos:
    frecuencias = pd.concat(
        [frecuencias,
         pd.DataFrame.from_dict(list_fdf[indice_doc], orient='index', columns=['TF' + str(indice_doc).zfill(4)])],
        axis=1)
frecuencias_documento_totales = frecuencias.loc[:, [x for x in frecuencias.columns if 'TF' in x]].sum().to_dict()
frecuencias_palabra_totales = frecuencias.sum(axis=1).to_dict()
for col in [x for x in frecuencias.columns if 'TF' in x]:
    frecuencias[col.replace('TF', 'WTF')] = frecuencias.loc[:, col].apply(
        lambda x: x / frecuencias_documento_totales[col])
frecuencias = frecuencias.fillna(0)

frecuencias['tmp'] = pd.DataFrame.from_dict(frecuencias_palabra_totales, orient='index').fillna(0)
columnas = frecuencias.columns
for col in [x for x in columnas if 'TF' in x and 'WTF' not in x]:
    frecuencias[col.replace('TF', 'WIDF')] = frecuencias.loc[:, col] / frecuencias.loc[:, 'tmp']
del frecuencias['tmp']

file_vocabulary = 'resultados/vocabulary.txt'
file_representacion = 'resultados/representacion'
file_inverso = 'resultados/fichero_inverso.txt'

# Escribimos los outputs
with open(file_vocabulary,'w') as f:
    f.write("\t".join(list(indice.index.values)))

for documentos in range(i):
    with open(file_representacion + 'WTF'+str(documentos).zfill(4)+'.txt', 'w') as f:
        f.write("\t".join([str(x) for x in frecuencias['WTF'+str(documentos).zfill(4)].values]))

for documentos in range(i):
    with open(file_representacion + 'WIDF'+str(documentos).zfill(4)+'.txt', 'w') as f:
        f.write("\t".join([str(x) for x in frecuencias['WIDF'+str(documentos).zfill(4)].values]))

columnas = [x for x in frecuencias.columns if 'TF' in x and 'WTF' not in x and 'WIDF' not in x]
with open(file_inverso,'w') as f:
    for linea in frecuencias.iterrows():
        linea = "\t".join([str(int(linea[1][col])) for col in columnas])
        f.write(linea)
sys.exit()
