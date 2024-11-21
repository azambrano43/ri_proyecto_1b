import pandas as pd
import os
import spacy
import re
import numpy as np
import gensim.downloader as api

from preprocessing_functions import preprocesar_contenido, obtener_stopwords, obtener_signos_puntuacion
from vectorization_functions import vectorize_bow, vectorize_tfidf, vectorize_word2vec
from search_functions import buscar_documentos


directorio_base = os.path.join(os.getcwd())  # Volver a directorio RI_PROYECTO_1B
cls = lambda: os.system("cls")


def main():

    print(directorio_base)
    df_corpus_procesado = pd.read_csv(f"{directorio_base}\data\corpus_procesado.csv")

    # Carga de stopwords y signos de puntuacion
    stopwords = obtener_stopwords("stopwords", directorio_base)
    signos_puntuacion = obtener_signos_puntuacion()

    # Aplicar la vectorización a todo el corpus
    print("Vectorizando corpus a BOW")
    X_bow, bow_vectorizer = vectorize_bow(df_corpus_procesado["contenido_preproc_str"])
    print("Vectorizando con TF-IDF")
    X_tfidf, tfidf_vectorizer = vectorize_tfidf(df_corpus_procesado["contenido_preproc_str"])
    print("Vectorizando con Word2Vec")
    X_word2vec, word2vec_vectorizer = vectorize_word2vec(df_corpus_procesado["contenido_preproc_str"])


    while(True):
        cls()

        print("BUSCADOR DE PRUEBA")
        print("tfidf -- Utilizar TF-IDF para búsqueda")
        print("bow -- Utilizar Bag Of Words para búsqueda")
        print("tfidf -- Utilizar Word2Vec para búsqueda")
        print("salir -- Salir del buscador")
        print("")
        metodo_selec = str(input("Escribe una de las opciones para comenzar:   "))

        if metodo_selec == "salir":
            return "Saliendo..."

        print("")
        consulta = str(input("A continuación, escriba su consulta:   "))

        consulta_preproc = preprocesar_contenido(consulta, stopwords, signos_puntuacion)

        if metodo_selec.lower() == "tfidf":
            corpus_vectorizado, vectorizador = X_tfidf, tfidf_vectorizer
        elif metodo_selec.lower() == "bow":
            corpus_vectorizado, vectorizador = X_bow, bow_vectorizer
        elif metodo_selec.lower() == "word2vec":
            corpus_vectorizado, vectorizador = X_word2vec, word2vec_vectorizer
        else:
            return "ERROR, METODO NO SOPORTADO"


        indices_ordenados, resultados = buscar_documentos(consulta_preproc, corpus_vectorizado, vectorizador, metodo=metodo_selec)

        # Preparar los resultados incluyendo índice, relevancia y texto
        resultados_detallados = [
            {
                "indice": indice,
                "relevancia": resultados[indice],
                "texto": df_corpus_procesado['contenido'][indice]  # Incluir el texto crudo del documento
            }
            for indice in indices_ordenados
        ]

        print("")
        print("MOSTRANDO RESULTADOS RELEVANTES")
        for resultado in resultados_detallados[:10]:
            print(f"Índice: {resultado['indice']}, Relevancia: {resultado['relevancia']:.2f}")
            print(f"Texto: {resultado['texto']}\n")


main()