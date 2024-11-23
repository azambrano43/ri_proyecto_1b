import pandas as pd
from collections import defaultdict

def construir_indice_invertido(df, directorio_base, texto_column='contenido_preproc_str', id_column='id'):
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
        
    path_guardado_ii = f"{directorio_base}\data\indice_invertido.txt"
    guardar_indice_invertido(indice_invertido, path_guardado_ii)
    
    return f"Índice invertido creado y guardado exitosamente en data\indice_invertido.txt"


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


def cargar_indice_invertido(input_path):
    """
    Carga el índice invertido desde un archivo de texto con formato legible.

    Parámetros:
    - input_path (str): Ruta del archivo de entrada.

    Retorno:
    - indice_invertido (dict): El índice invertido cargado desde el archivo.
    """
    indice_invertido = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        termino = None
        for line in f:
            line = line.strip()
            if line.startswith("Término:"):
                termino = line.split("Término:")[1].strip()
                indice_invertido[termino] = {}
            elif line.startswith("Documento:"):
                doc_info = line.split(",")
                doc_id = doc_info[0].split("Documento:")[1].strip()
                frecuencia = int(doc_info[1].split("Frecuencia:")[1].strip())
                indice_invertido[termino][doc_id] = frecuencia
            # Ignorar líneas vacías
    return indice_invertido