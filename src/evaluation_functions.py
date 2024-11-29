from sklearn.metrics import precision_score, recall_score, f1_score

def evaluar_resultados(resultados, relevantes):
    """
    Calcula precisión, recall y F1-score para una consulta dada.
    
    Parámetros:
    - resultados: Lista de documentos recuperados (cada uno con clave 'id' y 'relevancia').
    - relevantes: DataFrame de Pandas que contiene una columna 'id'.

    Retorna:
    - Diccionario con precisión, recall y F1-score.
    """
    # Lista de IDs relevantes
    ids_relevantes = set(relevantes['id'].tolist())
    
    # Filtrar resultados con relevancia diferente de 0
    resultados_filtrados = [doc for doc in resultados if doc.get('relevancia', 0) != 0]
    
    # Extraer los IDs de los resultados filtrados
    ids_recuperados = set(doc['id'] for doc in resultados_filtrados)
    
    # Calcular True Positives, False Positives y False Negatives
    tp = len(ids_recuperados & ids_relevantes)  # Intersección de relevantes y recuperados
    fp = len(ids_recuperados - ids_relevantes)  # Recuperados que no son relevantes
    fn = len(ids_relevantes - ids_recuperados)  # Relevantes que no fueron recuperados
    
    # Precisión, Recall y F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
