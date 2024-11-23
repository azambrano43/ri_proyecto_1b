import pandas as pd
import time
import os
import numpy as np
import gensim.downloader as api

from inverted_index import construir_indice_invertido, cargar_indice_invertido
from preprocessing_functions import preprocesar_contenido, obtener_stopwords, obtener_signos_puntuacion
from vectorization_functions import vectorize_bow, vectorize_tfidf, vectorize_word2vec
from search_functions import buscar_documentos, buscar_palabra_indice_invertido

max_resultados = 10
directorio_base = os.path.join(os.getcwd())  # Volver a directorio RI_PROYECTO_1B
cls = lambda: os.system("cls")


def main():

    # ======= Cargado de corpus como DataFrame y del índice invertido como diccionario =======
    df_corpus_procesado = pd.read_csv(f"{directorio_base}\data\corpus_procesado.csv")
    df_corpus_procesado["id"] = df_corpus_procesado["id"].astype("string")
    df_corpus_procesado["contenido"] = df_corpus_procesado["contenido"].astype("string")
    indice_invertido = cargar_indice_invertido(f"{directorio_base}\data\indice_invertido.txt")
    #print(indice_invertido)
    # ========================================================================================


    # ========== Habilitar solo si se quiere generar nuevamente el índice invertido ==========
    #construir_indice_invertido(df_corpus_procesado, directorio_base)
    # ========================================================================================


    # ==================== Carga de stopwords y signos de puntuacion =========================
    stopwords = obtener_stopwords("stopwords", directorio_base)
    signos_puntuacion = obtener_signos_puntuacion()
    # ========================================================================================


    # ==================== Aplicar técnicas de vectorización al corpus =======================
    print("Vectorizando corpus con BOW...")
    X_bow, bow_vectorizer = vectorize_bow(df_corpus_procesado["contenido_preproc_str"])

    print("Vectorizando corpus con TF-IDF...")
    X_tfidf, tfidf_vectorizer = vectorize_tfidf(df_corpus_procesado["contenido_preproc_str"])

    print("Vectorizando corpus con Word2Vec...")
    w2v_model = api.load("word2vec-google-news-300")
    X_word2vec = vectorize_word2vec(df_corpus_procesado["contenido_preproc_str"], w2v_model)

    cls()
    # ========================================================================================


    while(True):
        print("\n\n")
        print("|============== BUSCADOR DE PRUEBA ==============| \n")
        print("1 --> Utilizar TF-IDF para búsqueda")
        print("2 --> Utilizar Bag Of Words para búsqueda")
        print("3 --> Utilizar Word2Vec para búsqueda")
        print("4 --> Utilizar índice invertido")
        print("0 --> Salir del buscador")
        print("")
        print("Nota --> Indice invertido debe recibir una única palabra para buscar")
        print("Si no es el caso, se usará la primera palabra válida de la oración")

        metodo_selec = str(input("\nEscribe una de las opciones para comenzar:   "))

        cls()

        if metodo_selec == "1":
            corpus_vectorizado, vectorizador = X_tfidf, tfidf_vectorizer
        elif metodo_selec == "2":
            corpus_vectorizado, vectorizador = X_bow, bow_vectorizer
        elif metodo_selec == "3":
            corpus_vectorizado, vectorizador = X_word2vec, w2v_model
        elif metodo_selec == "4":
            pass
        elif metodo_selec == "0":
            return "Saliendo..."
        else:
            print("Escoja un método válido")
            break
        
        print("")
        consulta = str(input("A continuación, escriba su consulta:   "))

        # Preprocesar la consulta con las funciones de preprocessing_functions
        consulta_preproc = preprocesar_contenido(consulta, stopwords, signos_puntuacion)

        if metodo_selec == "1" or metodo_selec == "2" or metodo_selec == "3":
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
            print("")
            print(f"Mostrando resultados relevantes para la búsqueda:   {consulta}")
            print("")
            for resultado in resultados_detallados[:max_resultados]:
                print(f"Índice: {resultado['indice']}, Relevancia: {resultado['relevancia']:.3f}")
                print(f"Texto: {resultado['texto']}\n")

        elif metodo_selec == "4":
            if len(consulta_preproc.split()) == 1:
                resultados = buscar_palabra_indice_invertido(indice_invertido, df_corpus_procesado, consulta_preproc, top_n=max_resultados)
            elif len(consulta_preproc.split()) > 1:
                print("Búsqueda por índice invertido ha detectado más de una palabra, usando la primera válida...")
                resultados = buscar_palabra_indice_invertido(indice_invertido, df_corpus_procesado, consulta_preproc.split()[0], top_n=max_resultados)
            else:
                print("La palabra es una stop-word o no se encuentra en el índice invertido...\n\n")
                time.sleep(2)
                pass

            # Mostrar los resultados
            for contenido, doc_id in resultados:
                print(f"Documento con ID: {doc_id}:\n{contenido}\n")

        str(input("|============== Presione enter para continuar ==============|"))

        cls()


main()