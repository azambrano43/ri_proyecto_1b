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
│   ├── corpus_training_procesado.csv # Corpus del training original ya procesado
│   ├── corpus_training.csv     # Corpus del training original sin procesar
│   ├── stopwords/              # Lista de palabras vacías para preprocesamiento
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
│   ├── evaluation_functions.py    # Funciones para calcular las metricas de los resultados
│   └── vectorization_functions.py # Funciones para la vectorización de texto
│
├── templates/                 # Templates para la pagina web
│   └── index.html             # template de pagina de busqueda
│   └── resultados.html        # template de pagina de resultados
│
├── static/                 # Estilos para la pagina web
│   └── index.css             # estilos de pagina de busqueda
│   └── resultados.css        # estilos de pagina de resultados
│
├── index.py                  # Archivo .py encargado del manejo de la logica web
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

## Requisitos

Instalar las siguientes dependencias antes de ejecutar el proyecto:

```bash
pip install scikit-learn gensim pandas numpy flask nltk
```

### Descarga de recursos de NLTK

El proyecto utiliza recursos de NLTK para el preprocesamiento del texto. Ejecute el siguiente código una vez para descargarlos:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_en')
nltk.download('wordnet')
```

## Ejecución
### Clona este repositorio:

```bash
git clone https://github.com/azambrano43/ri_proyecto_1b.git
cd ri_proyecto_1b
```

### Abre tu navegador en http://127.0.0.1:5000 para acceder a la aplicación.

### Desde el cuaderno interactivo

1. Navegue a la carpeta `notebooks/`.  
2. Abra y ejecute el archivo `Proyecto_RI_1B.ipynb`.

### Desde la línea de comandos

1. Ejecute el archivo principal `main.py`:

```bash
python src/main.py
```

### Desde interfaz grafica

1. Ejecute el archivo principal `index.py`:

```bash
python index.py
```

## Motor de Búsqueda Basado en Flask  

Este proyecto implementa un motor de búsqueda con una interfaz de usuario funcional que permite realizar consultas a través del navegador utilizando diferentes técnicas de recuperación de información. A continuación, se detallan las tecnologías utilizadas, la estructura del proyecto y cómo se abordan los objetivos planteados.

---

### 1. **Integración de la Interfaz de Usuario utilizando Flask (HTML + CSS + JavaScript)**

#### Tecnologías Utilizadas
- **Flask**: Framework para el desarrollo del backend y el enrutamiento de la aplicación web.
- **HTML y CSS**: Para diseñar y estructurar las páginas web con un enfoque en la usabilidad.
- **JavaScript**: Para implementar funcionalidades interactivas, como el manejo dinámico de los resultados y la interacción con el usuario.

#### Implementación
- **Página de Inicio (`index.html`)**:  
  - Contiene una barra de búsqueda donde el usuario puede ingresar su consulta y seleccionar el método de búsqueda deseado.
  - Los métodos disponibles son: *TF-IDF*, *Bag of Words (BoW)*, *Word2Vec* e *Índice Invertido*.  
- **Página de Resultados (`resultados.html`)**:  
  - Muestra los documentos recuperados paginados.  
  - Ofrece opciones para navegar entre páginas y ver más o menos contenido de cada resultado.  
  - Se integraron indicadores visuales para mostrar métricas de evaluación (precisión, recall y F1-score) cuando están disponibles.  
- **Estilización**:  
  - Los estilos están definidos en archivos CSS dentro de la carpeta `static/`, mejorando la experiencia visual del usuario.
  - Se añadió un diseño adaptativo para asegurar la compatibilidad con dispositivos móviles.

---

### 2. **Interacción del Usuario con el Motor de Búsqueda**

#### Tecnologías Utilizadas
- **Flask**: Para recibir y procesar las consultas del usuario.  
- **Python**: Para la lógica del motor de búsqueda y preprocesamiento.  
- **Librerías de Machine Learning**:  
  - `scikit-learn`: Para calcular similitudes con TF-IDF y Bag of Words.  
  - `gensim`: Para el modelo Word2Vec.  
  - Módulos personalizados para la carga de índices invertidos y vectorización.

#### Funcionalidad
- **Consulta desde el navegador**:  
  - El usuario ingresa una consulta en lenguaje natural, que se procesa mediante técnicas de normalización, eliminación de stopwords y reducción de signos de puntuación.
- **Métodos de Búsqueda**:  
  - **TF-IDF**: Calcula la relevancia basándose en la frecuencia de palabras y su importancia en el corpus.  
  - **Bag of Words (BoW)**: Representa los documentos como vectores binarios o ponderados.  
  - **Word2Vec**: Utiliza un modelo preentrenado para evaluar similitudes semánticas.  
  - **Índice Invertido**: Recupera documentos basados en palabras clave específicas.
- **Paginación**:  
  - Los resultados se dividen en bloques definidos por el parámetro `resultados_por_pagina`.  
  - Se incluye navegación para ir a páginas específicas.  
- **Alertas y mensajes**:  
  - El sistema muestra notificaciones cuando no se encuentran resultados o cuando ciertas configuraciones no son aplicables, como en consultas de múltiples palabras con índice invertido.
---

### 3. **Documentación del Proceso de Evaluación y Métricas de Desempeño**

#### Evaluación
Se desarrolló un módulo específico para calcular las métricas clave de recuperación de información:  
- **Precisión (Precision)**: Proporción de documentos relevantes en los resultados recuperados.  
- **Recall**: Proporción de documentos relevantes que fueron correctamente recuperados.  
- **F1-Score**: Métrica que combina precisión y recall para ofrecer un balance entre ambas.
**Nota importante:** Es importante resaltar que para la obtención de documentos
relevantes para realizar las métricas se utilizó los documentos de la carpeta training esto
puede afectar las métricas ya que para los documentos que se recuperan continente
tanto la carpeta test como training

#### Implementación
- **Evaluación Automática**:  
  - Los resultados del motor de búsqueda se comparan con un conjunto de datos relevantes predefinido para calcular las métricas.  
  - Los valores calculados se muestran en la página de resultados junto con indicadores visuales como barras de progreso.  
- **Indicadores Visuales**:  
  - Se utilizan barras dinámicas que indican el porcentaje de precisión, recall y F1-score.
  - Si no es posible calcular métricas (por ejemplo, en el índice invertido), el sistema muestra un mensaje adecuado.
