from sklearn.metrics.pairwise import cosine_similarity
from vectorization_functions import vectorize_word2vec
import pandas as pd

def buscar_documentos(consulta, corpus_vectorizado, vectorizador, metodo="tfidf"):
    """
    Busca documentos en el corpus y devuelve los más relevantes según el método elegido (BoW, TF-IDF, Word2Vec).
    """
    # Vectorizar la consulta
    if metodo == "1":
        consulta_vec = vectorizador.transform([consulta])
    elif metodo == "2":
        consulta_vec = vectorizador.transform([consulta])
    elif metodo == "3":
        consulta_vec = vectorize_word2vec([consulta], vectorizador)[0].reshape(1, -1)
    else:
        raise ValueError("Método no soportado")
    
    matriz = corpus_vectorizado

    # Calcular similitudes
    similitudes = cosine_similarity(consulta_vec, matriz)
    resultados = similitudes[0]
    
    # Rankear documentos por relevancia y tomar los top_n
    indices_ordenados = resultados.argsort()[::-1]
    
    return indices_ordenados, resultados

def buscar_palabra_indice_invertido(indice_invertido, df, termino, top_n=10, texto_column='contenido'):
    """
    Busca un término en el índice invertido y devuelve los 'top_n' documentos más relevantes.
    Solo muestra el contenido del documento.
    
    Parámetros:
    - indice_invertido (dict): El índice invertido construido.
    - df (pd.DataFrame): DataFrame que contiene los documentos con sus contenidos.
    - termino (str): El término a buscar en el índice invertido.
    - top_n (int): Número de resultados más relevantes a devolver.
    - texto_column (str): Columna que contiene el contenido del documento.
    
    Retorno:
    - lista_resultados (list): Lista de los documentos más relevantes con su contenido.
    """
    # Verificar si el término está en el índice invertido
    if termino in indice_invertido:
        # Obtener los documentos con el término y su frecuencia
        documentos = indice_invertido[termino]
        
        # Ordenar los documentos por frecuencia y tomar los 'top_n' más relevantes
        documentos_relevantes = sorted(documentos.items(), key=lambda item: item[1], reverse=True)[:top_n]
        #print(documentos_relevantes)
        
        # Extraer el contenido de los documentos relevantes
        lista_resultados = []
        for doc_id, _ in documentos_relevantes:
            # Obtener el contenido del documento desde el DataFrame
            contenido = df[df['id'] == doc_id][texto_column].values[0]
            lista_resultados.append([contenido, doc_id])
        
        return lista_resultados
    else:
        return []