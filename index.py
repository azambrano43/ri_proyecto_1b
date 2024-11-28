# Importación de librerías y módulos necesarios
from flask import Flask, render_template, request, jsonify  # Framework Flask para la web
import pandas as pd  # Manipulación y análisis de datos
import os  # Operaciones del sistema operativo
import sys  # Interacción con el sistema
from sklearn.metrics.pairwise import cosine_similarity  # Para calcular similitud entre vectores

# Agregar la carpeta src al PYTHONPATH para importar módulos personalizados
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Importación de librerías y funciones personalizadas
import gensim.downloader as api  # Carga de modelos preentrenados de Word2Vec
from inverted_index import cargar_indice_invertido  # Carga de índices invertidos
from preprocessing_functions import (
    obtener_stopwords, 
    obtener_signos_puntuacion, 
    preprocesar_contenido
)  # Preprocesamiento del texto
from vectorization_functions import (
    vectorize_bow, 
    vectorize_tfidf, 
    vectorize_word2vec
)  # Métodos de vectorización
from search_functions import buscar_documentos, buscar_palabra_indice_invertido  # Búsqueda
from evaluation_functions import evaluar_resultados  # Evaluación de resultados

# Inicialización de la aplicación Flask
app = Flask(__name__)

# ====== Configuración inicial ======

# Definición del directorio base
directorio_base = os.path.join(os.getcwd())

# Carga del corpus preprocesado y los datos relevantes
df_corpus_procesado = pd.read_csv(f"{directorio_base}/data/corpus_procesado.csv")
relevantes = pd.read_csv(f"{directorio_base}/data/corpus_training_procesado.csv")

# Configuración de tipos de datos para garantizar consistencia
df_corpus_procesado["id"] = df_corpus_procesado["id"].astype("string")
df_corpus_procesado["contenido"] = df_corpus_procesado["contenido"].astype("string")

# Carga del índice invertido
indice_invertido = cargar_indice_invertido(f"{directorio_base}/data/indice_invertido.txt")
relevantes["id"] = relevantes["id"]

# Obtención de stopwords y signos de puntuación
stopwords = obtener_stopwords("stopwords", directorio_base)
signos_puntuacion = obtener_signos_puntuacion()

# Vectorización de los documentos usando diferentes métodos
X_bow, bow_vectorizer = vectorize_bow(df_corpus_procesado["contenido_preproc_str"])  # Bag of Words
X_tfidf, tfidf_vectorizer = vectorize_tfidf(df_corpus_procesado["contenido_preproc_str"])  # TF-IDF
w2v_model = api.load("word2vec-google-news-300")  # Modelo Word2Vec preentrenado
X_word2vec = vectorize_word2vec(df_corpus_procesado["contenido_preproc_str"], w2v_model)  # Word2Vec

# Configuración de resultados por página
resultados_por_pagina = 10

# ====== Rutas ======

# Ruta principal para la página de inicio
@app.route("/")
def home():
    return render_template("index.html")

# Ruta para manejar las búsquedas
@app.route("/buscar", methods=["GET", "POST"])
def buscar():
    if request.method == "POST":  # Si la consulta se envía mediante un formulario
        consulta = request.form.get("consulta")  # Obtener la consulta ingresada
        metodo = request.form.get("metodo")  # Método seleccionado para la búsqueda
        raw_consulta = consulta  # Almacenar la consulta original
    else:  # Si la consulta se pasa como parámetro en la URL
        consulta = request.args.get("consulta")
        metodo = request.args.get("metodo")
        raw_consulta = consulta

    # Variables para métricas y mensajes
    mensaje = None
    precision, recall, f1_score = None, None, None

    # Preprocesar la consulta
    consulta = preprocesar_contenido(consulta, stopwords, signos_puntuacion)
    
    # Manejo de diferentes métodos de búsqueda
    if metodo == "indice_invertido":
        if len(consulta.split()) == 1:  # Consulta con una sola palabra
            resultados = buscar_palabra_indice_invertido(
                indice_invertido, df_corpus_procesado, consulta, top_n=None
            )
        elif len(consulta.split()) > 1:  # Consulta con varias palabras
            mensaje = "Búsqueda por índice invertido: más de una palabra detectada, se usará la primera."
            resultados = buscar_palabra_indice_invertido(
                indice_invertido, df_corpus_procesado, consulta.split()[0], top_n=None
            )
    elif metodo == "tfidf":
        # Búsqueda utilizando el método TF-IDF
        indices, similitudes = buscar_documentos(
            consulta, X_tfidf, tfidf_vectorizer, metodo="tfidf"
        )
    elif metodo == "bow":
        # Búsqueda utilizando el método Bag of Words
        indices, similitudes = buscar_documentos(
            consulta, X_bow, bow_vectorizer, metodo="bow"
        )
    elif metodo == "word2vec":
        # Búsqueda utilizando Word2Vec
        indices, similitudes = buscar_documentos(
            consulta, X_word2vec, w2v_model, metodo="word2vec"
        )
    else:
        return jsonify({"error": "Método no soportado."}), 400

    # Construcción de resultados para métodos diferentes al índice invertido
    if metodo != "indice_invertido":
        resultados = [
            {
                "id": int(indice),
                "contenido": df_corpus_procesado['contenido'][indice],
                "relevancia": float(similitudes[indice]),
            }
            for indice in indices
        ]
        if resultados:
            # Evaluación de los resultados utilizando métricas como precisión, recall y F1
            evaluacion = evaluar_resultados(resultados, relevantes)
            precision = evaluacion["precision"]
            recall = evaluacion["recall"]
            f1_score = evaluacion["f1_score"]
    else:
        # Resultados para índice invertido (sin calcular relevancia)
        resultados = [
            {
                "id": int(doc_id),
                "contenido": contenido,
                "relevancia": None,
            }
            for contenido, doc_id in resultados
        ]

    # Paginación de resultados
    pagina = int(request.args.get("pagina", 1))
    inicio = (pagina - 1) * resultados_por_pagina
    fin = inicio + resultados_por_pagina
    resultados_paginados = resultados[inicio:fin]

    # Total de páginas para la navegación
    total_paginas = (len(resultados) + resultados_por_pagina - 1) // resultados_por_pagina

    # Renderizado de la plantilla con los resultados
    return render_template(
        "resultados.html",
        raw_consulta=raw_consulta,
        consulta=consulta,
        metodo=metodo,
        resultados=resultados_paginados,
        pagina=pagina,
        total_paginas=total_paginas,
        mensaje=mensaje,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )

# ====== Punto de entrada ======
if __name__ == "__main__":
    app.run(debug=True)  # Ejecutar la aplicación en modo de depuración
