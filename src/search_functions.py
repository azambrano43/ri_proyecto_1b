from sklearn.metrics.pairwise import cosine_similarity
from vectorization_functions import vectorize_word2vec

def buscar_documentos(consulta, corpus_vectorizado, vectorizador, metodo="tfidf"):
    """
    Busca documentos en el corpus y devuelve los más relevantes según el método elegido (BoW, TF-IDF, Word2Vec).
    """
    # Vectorizar la consulta
    if metodo == "bow":
        consulta_vec = vectorizador.transform([consulta])
    elif metodo == "tfidf":
        consulta_vec = vectorizador.transform([consulta])
    elif metodo == "word2vec":
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