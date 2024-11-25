from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

# Agregar la carpeta src al PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gensim.downloader as api
from inverted_index import cargar_indice_invertido
from preprocessing_functions import obtener_stopwords, obtener_signos_puntuacion, preprocesar_contenido
from vectorization_functions import vectorize_bow, vectorize_tfidf, vectorize_word2vec
from search_functions import buscar_documentos, buscar_palabra_indice_invertido

app = Flask(__name__)

# ====== Configuración inicial ======
directorio_base = os.path.join(os.getcwd())
df_corpus_procesado = pd.read_csv(f"{directorio_base}/data/corpus_procesado.csv")
df_corpus_procesado["id"] = df_corpus_procesado["id"].astype("string")
df_corpus_procesado["contenido"] = df_corpus_procesado["contenido"].astype("string")
indice_invertido = cargar_indice_invertido(f"{directorio_base}/data/indice_invertido.txt")

stopwords = obtener_stopwords("stopwords", directorio_base)
signos_puntuacion = obtener_signos_puntuacion()

X_bow, bow_vectorizer = vectorize_bow(df_corpus_procesado["contenido_preproc_str"])
X_tfidf, tfidf_vectorizer = vectorize_tfidf(df_corpus_procesado["contenido_preproc_str"])
w2v_model = api.load("word2vec-google-news-300")
X_word2vec = vectorize_word2vec(df_corpus_procesado["contenido_preproc_str"], w2v_model)

resultados_por_pagina = 10

# ====== Rutas ======

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/buscar", methods=["GET", "POST"])
def buscar():
    if request.method == "POST":
        consulta = request.form.get("consulta")
        metodo = request.form.get("metodo")
        raw_consulta = consulta
    else:
        consulta = request.args.get("consulta")
        metodo = request.args.get("metodo")
        raw_consulta = consulta
    consulta = preprocesar_contenido(consulta, stopwords, signos_puntuacion)
    
    #if metodo == "indice_invertido":
    #    if len(consulta.split()) == 1:
    #            resultados = buscar_palabra_indice_invertido(indice_invertido, df_corpus_procesado, consulta, top_n=max_resultados)
    #    elif len(consulta.split()) > 1:
    #            print("Búsqueda por índice invertido ha detectado más de una palabra, usando la primera válida...")
    #            resultados = buscar_palabra_indice_invertido(indice_invertido, df_corpus_procesado, consulta.split()[0], top_n=max_resultados)
    
    if metodo == "tfidf":
            indices, similitudes = buscar_documentos(
                consulta, X_tfidf, tfidf_vectorizer, metodo="tfidf"
            )
    elif metodo == "bow":
            indices, similitudes = buscar_documentos(
                consulta, X_bow, bow_vectorizer, metodo="bow"
            )
    elif metodo == "word2vec":
            indices, similitudes = buscar_documentos(
                consulta, X_word2vec, w2v_model, metodo="word2vec"
            )
    else:
            return jsonify({"error": "Método no soportado."}), 400

    resultados = [
            {
                "id": int(indice),
                "contenido": df_corpus_procesado['contenido'][indice],
                "relevancia": float(similitudes[indice]),

            }
            for indice in indices
        ]
    
    pagina = int(request.args.get("pagina", 1))
    inicio = (pagina - 1) * resultados_por_pagina
    fin = inicio + resultados_por_pagina
    resultados_paginados = resultados[inicio:fin]

    total_paginas = (len(resultados) + resultados_por_pagina - 1) // resultados_por_pagina

    return render_template(
        "resultados.html",
        raw_consulta=raw_consulta,
        consulta=consulta,
        metodo=metodo,
        resultados=resultados_paginados,
        pagina=pagina,
        total_paginas=total_paginas,
    )

# ====== Punto de entrada ======
if __name__ == "__main__":
    app.run(debug=True)
