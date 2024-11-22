import re
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# Función para convertir etiquetas POS de nltk a WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Por defecto

# Función de preprocesamiento
def preprocesar_contenido(contenido, stopwords, signos_puntuacion):

    # Definir el lematizador
    lemmatizer = WordNetLemmatizer()

    # Convertir texto a minúsculas
    contenido = contenido.lower()

    # Reemplazar caracteres y limpiar texto
    contenido = contenido.replace("'", " ").replace(" &lt;", " ").replace("&lt;", " ")
    contenido = contenido.replace("trading", "trade").replace("united states of america", "usa").replace("united states", "usa")
    contenido = re.sub(r'\d+', '', contenido)  # Eliminar números
    contenido = re.sub(r'\s+', ' ', contenido)  # Eliminar saltos de línea y espacios redundantes

    # Remover signos de puntuación
    for signo in signos_puntuacion:
        contenido = contenido.replace(signo, "")

    # Tokenización
    contenido_tokenizado = contenido.split()

    # Eliminar stopwords
    tokens_sin_stopwords = [word for word in contenido_tokenizado if word not in stopwords]

    # Etiquetado POS (Part Of Speech)
    tokens_pos = pos_tag(tokens_sin_stopwords)

    # Lematización utilizando etiquetado POS
    tokens_lematizados = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in tokens_pos
    ]

    # Reconstruir texto lematizado a un solo string
    contenido_lematizado_str = " ".join(tokens_lematizados)

    return contenido_lematizado_str

def obtener_stopwords(archivo_stopwords, path_base):
    # Obtencion de las stopwords desde el archivo 'stopwords'
 
    with open(f"{path_base}\data\{archivo_stopwords}", mode='r', encoding='utf-8') as file:
        contenido_stopwords = file.read()

    stopwords = []

    for words in contenido_stopwords.split("\n"):
        stopword = words.replace("\n", "")
        stopwords.append(stopword)

    return stopwords

def obtener_signos_puntuacion():
    signos_puntuacion = [",", ".", ":", "...", "-", "_", "+", ";", '"', "(", ")", "[", "]", "%", "$", "#", "@", "&", "?", "!", "/", ">", "<"]
    return signos_puntuacion