import pandas as pd
from collections import defaultdict

def construir_indice_invertido(df, texto_column='contenido_preproc_str', id_column='id'):
    """
    Construye un índice invertido a partir de un DataFrame con textos preprocesados.
    Solo almacena palabras con longitud >= 3 y ordena los términos por frecuencia descendente.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame que contiene los textos.
    - texto_column (str): Nombre de la columna que contiene los textos preprocesados.
    - id_column (str): Nombre de la columna que contiene los IDs de los documentos.
    
    Retorno:
    - indice_invertido (dict): Diccionario donde las claves son términos y los valores son diccionarios 
      con el ID del documento y la frecuencia del término.
    """
    # Estructura del índice invertido
    indice_invertido = defaultdict(lambda: defaultdict(int))
    
    # Construir el índice invertido
    for _, row in df.iterrows():
        doc_id = row[id_column]
        texto = row[texto_column]
        for token in texto.split():
            if len(token) >= 3:  # Filtrar palabras con longitud >= 3
                indice_invertido[token][doc_id] += 1  # Incrementar frecuencia
    
    # Ordenar términos por frecuencia descendente dentro de cada documento
    for termino in indice_invertido:
        indice_invertido[termino] = dict(sorted(indice_invertido[termino].items(), 
                                                key=lambda item: item[1], reverse=True))
    
    return dict(indice_invertido)

def guardar_indice_invertido(indice_invertido, output_path):
    """
    Guarda el índice invertido en un archivo de texto con formato legible.
    
    Parámetros:
    - indice_invertido (dict): Índice invertido a guardar.
    - output_path (str): Ruta del archivo de salida.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for termino, documentos in sorted(indice_invertido.items()):  # Ordenar términos alfabéticamente
            f.write(f"Término: {termino}\n")
            for doc_id, frecuencia in documentos.items():  # Ya ordenados por frecuencia descendente
                f.write(f"    Documento: {doc_id}, Frecuencia: {frecuencia}\n")
            f.write("\n")  # Línea en blanco entre términos