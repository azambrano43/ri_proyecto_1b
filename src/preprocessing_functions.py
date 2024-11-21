import spacy
import re

nlp = spacy.load('en_core_web_sm')

def preprocesar_contenido(contenido, stopwords, signos_puntuacion):

    contenido = contenido.lower() # Texto en minúsculas

    contenido = contenido.replace("'", " ").replace(" &lt;", " ").replace("&lt;", " ")
    contenido = re.sub(r'\d+', '', contenido) # Eliminar números
    contenido = re.sub(r'\s+', ' ', contenido) # Eliminar saltos de línea y espacios redundantes

    for signo in signos_puntuacion:
        contenido = contenido.replace(signo, "")
    
    contenido_tokenizado = contenido.split()

    tokens = [word for word in contenido_tokenizado if word not in stopwords]

    contenido_nlp = nlp(" ".join(tokens))
    tokens_lematizados = [token.lemma_ for token in contenido_nlp]

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