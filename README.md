# Proyecto de Sistema de Recuperación de Información (SRI)

Este repositorio contiene el código y la estructura para el desarrollo de un Sistema de Recuperación de Información (SRI) basado en Python, utilizando Flask para la interfaz web. El objetivo principal es implementar un motor de búsqueda eficiente capaz de procesar y recuperar documentos relevantes dentro de un corpus predefinido.

---

## Estructura del Proyecto

```plaintext
RI_PROYECTO_1B/
│
├── data/                      # Datos utilizados en el proyecto
│   ├── test/                  # Datos de prueba
│   ├── training/              # Datos de entrenamiento
│   ├── cats.txt               # Categorías del corpus
│   ├── corpus_procesado.csv   # Corpus ya procesado
│   ├── corpus_sin_procesar.csv # Corpus original sin procesar
│   └── indice_invertido.txt   # Índice invertido generado
│
├── notebooks/                 # Notebooks de Jupyter para pruebas y análisis
│   └── pruebas.ipynb          # Cuaderno de pruebas y validación
│
├── src/                       # Código fuente principal del proyecto
│   ├── __pycache__/           # Archivos compilados automáticamente
│   ├── inverted_index.py      # Funciones para la creación y gestión del índice invertido
│   ├── main.py                # Punto de entrada principal para ejecutar el proyecto
│   ├── preprocessing_functions.py # Funciones para el preprocesamiento del texto
│   ├── search_functions.py    # Funciones relacionadas con la búsqueda en el índice
│   └── vectorization_functions.py # Funciones para la vectorización de texto
│
├── stopwords/                 # Lista de palabras vacías para preprocesamiento
├── README.md                  # Descripción general del proyecto
├── .gitignore                 # Archivos y carpetas ignorados por Git
```


## Descripción General

### Fases del Proyecto

#### 1. Adquisición de Datos
Consiste en la recopilación de documentos para conformar el corpus de texto inicial.

#### 2. Preprocesamiento del Texto
- Incluye tokenización, normalización, eliminación de stopwords y lematización.  
- El corpus procesado se almacena en el archivo `corpus_procesado.csv`.

#### 3. Vectorización
Representación de los documentos en espacios vectoriales mediante técnicas como Bag of Words (BoW), TF-IDF y Word2Vec.

#### 4. Indexación
Construcción de un índice invertido, almacenado en el archivo `indice_invertido.txt`.

#### 5. Motor de Búsqueda
Recuperación eficiente de documentos relevantes en base a consultas de usuario.

#### 6. Evaluación del Sistema
Medición del desempeño mediante métricas como precisión y recall.

---

### Asignaciones

- **Andrés Zambrano**: Desarrollo de las fases 2.1 a 2.4 (Adquisición, Preprocesamiento, Vectorización, Indexación).  
- **Dilan y David**: Implementación del Motor de Búsqueda y la Evaluación (Fases 2.5 a 2.6).

---

### Estado Actual

El proyecto se encuentra en la etapa de pruebas. Los módulos principales están siendo validados mediante el cuaderno `pruebas.ipynb` en la carpeta `notebooks`.  
También es posible ejecutar el archivo `main.py` desde la carpeta `src` para realizar pruebas integradas.

En próximas etapas se integrará una interfaz web con **Flask** para proporcionar una experiencia interactiva al usuario.

---

## Requisitos

Instalar las siguientes dependencias antes de ejecutar el proyecto:

```bash
pip install scikit-learn gensim pandas numpy flask
```

### Descarga de recursos de NLTK

El proyecto utiliza recursos de NLTK para el preprocesamiento del texto. Ejecute el siguiente código una vez para descargarlos:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

## Ejecución

### Desde el cuaderno interactivo

1. Navegue a la carpeta `notebooks/`.  
2. Abra y ejecute el archivo `pruebas.ipynb`.

### Desde la línea de comandos

1. Ejecute el archivo principal `main.py`:

```bash
python src/main.py
```

Esto procesará el corpus, generará índices invertidos y permitirá ejecutar búsquedas básicas.

## Próximos pasos

1. Integrar la interfaz de usuario utilizando Flask (HTML + CSS + JavaScript).  
2. Permitir al usuario interactuar con el motor de búsqueda a través de consultas desde el navegador.  
3. Documentar completamente el proceso de evaluación y métricas de desempeño.
