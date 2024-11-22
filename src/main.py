import pandas as pd
import os
import spacy
import re
import time
import numpy as np
import gensim.downloader as api

from inverted_index import construir_indice_invertido, guardar_indice_invertido
from preprocessing_functions import preprocesar_contenido, obtener_stopwords, obtener_signos_puntuacion
from vectorization_functions import vectorize_bow, vectorize_tfidf, vectorize_word2vec
from search_functions import buscar_documentos


directorio_base = os.path.join(os.getcwd())  # Volver a directorio RI_PROYECTO_1B
cls = lambda: os.system("cls")


def main():

    # Carga de corpus preprocesado
    df_corpus_procesado = pd.read_csv(f"{directorio_base}\data\corpus_procesado.csv")

    # Carga de stopwords y signos de puntuacion
    stopwords = obtener_stopwords("stopwords", directorio_base)
    signos_puntuacion = obtener_signos_puntuacion()

    # Aplicar la vectorización a todo el corpus
    print("Vectorizando corpus con BOW...")
    X_bow, bow_vectorizer = vectorize_bow(df_corpus_procesado["contenido_preproc_str"])

    print("Vectorizando corpus con TF-IDF...")
    X_tfidf, tfidf_vectorizer = vectorize_tfidf(df_corpus_procesado["contenido_preproc_str"])

    print("Vectorizando corpus con Word2Vec...")
    w2v_model = api.load("word2vec-google-news-300")
    X_word2vec = vectorize_word2vec(df_corpus_procesado["contenido_preproc_str"], w2v_model)

    # Creación del índice invertido en /data/indice_invertido.txt
    indice_invertido = construir_indice_invertido(df_corpus_procesado)

    # Guardar el índice invertido en un archivo
    path_indice_invertido = f"{directorio_base}\indice_invertido.txt"
    guardar_indice_invertido(indice_invertido, path_indice_invertido)

    print(f"Índice invertido creado y almacenado en indice_invertido.txt")

    cls()


    while(True):
        print("\n\n")
        print("|============== BUSCADOR DE PRUEBA ==============| \n")
        print("tfidf -- Utilizar TF-IDF para búsqueda")
        print("bow -- Utilizar Bag Of Words para búsqueda")
        print("word2vec -- Utilizar Word2Vec para búsqueda")
        print("")
        print("salir -- Salir del buscador")
        print("")

        metodo_selec = str(input("Escribe una de las opciones para comenzar:   "))

        cls()

        if metodo_selec.lower() == "tfidf":
            corpus_vectorizado, vectorizador = X_tfidf, tfidf_vectorizer
        elif metodo_selec.lower() == "bow":
            corpus_vectorizado, vectorizador = X_bow, bow_vectorizer
        elif metodo_selec.lower() == "word2vec":
            corpus_vectorizado, vectorizador = X_word2vec, w2v_model
        elif metodo_selec.lower() == "salir":
            return "Saliendo..."
        else:
            print("Escoja un método válido")
            break
        
        print("")
        consulta = str(input("A continuación, escriba su consulta:   "))

        # Preprocesar la consulta con las funciones de preprocessing_functions
        consulta_preproc = preprocesar_contenido(consulta, stopwords, signos_puntuacion)

        # Realizar la busqueda y obtener resultados relevantes con search_functions
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

        
        cls()
        print("")
        print(f"Mostrando resultados relevantes para la búsqueda:   {consulta}")
        print("")
        for resultado in resultados_detallados[:10]:
            print(f"Índice: {resultado['indice']}, Relevancia: {resultado['relevancia']:.3f}")
            print(f"Texto: {resultado['texto']}\n")

        str(input("|============== Presione enter para continuar ==============|"))


main()