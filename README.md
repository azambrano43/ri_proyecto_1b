# Proyecto de Sistema de Recuperación de Información (SRI)

Este repositorio contiene el código y la estructura del proyecto de un sistema de recuperación de información (SRI) desarrollado en Python con Flask.

## Estructura del Proyecto

```plaintext
proyecto_SRI/
│
├── src/                  # Código fuente principal del proyecto
│   ├── __init__.py           # Inicialización del módulo src
│   ├── data_acquisition.py   # Adquisición de datos (Paso 2.1)
│   ├── preprocessing.py      # Preprocesamiento del texto (Paso 2.2)
│   ├── vectorization.py      # Representación de datos en espacio vectorial (Paso 2.3)
│   ├── indexing.py           # Indexación (Paso 2.4)
│   ├── search_engine.py      # Motor de búsqueda (Paso 2.5)
│   ├── evaluation.py         # Evaluación del sistema (Paso 2.6)
│   ├── app.py                # Archivo principal de Flask para iniciar la aplicación
│   ├── templates/            # Plantillas HTML para la interfaz web
│   │   └── index.html        # Página principal con la interfaz de búsqueda
│   └── static/               # Archivos estáticos (CSS, JS) para la interfaz
│       ├── style.css         # Estilos CSS para la interfaz web
│       └── app.js            # Lógica JavaScript para manejar eventos en la interfaz
│
├── data/                 # Carpeta para almacenar el corpus descargado y otros datos procesados
├── docs/                 # Documentación de cada fase del proyecto
├── notebooks/            # Notebooks de Jupyter para pruebas y análisis exploratorio
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Descripción general del proyecto
```

Asignaciones:
- Andrés --> Desde el punto 2.1 hasta el 2.4
- Dilan y David --> Punto 2.5 hasta el 2.7

En este proyecto se va a utilizar python con javascript, mediante el uso de Flask.