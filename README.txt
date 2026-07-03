================================================================================
                            PROJECT README
================================================================================

		----------------------------------------
			TRABAJO DE FIN DE MÁSTER
		----------------------------------------

TÍTULO: Ontologías emocionales y modelos de lenguaje: Hacia una representación
	semántico-afectiva del conocimiento

INSTITUCIÓN: Escuela Técnica Superior de Ingeniería Informática,
	     Universidad de Sevilla

TITULACIÓN: Máster en Lógica, Computación e Inteligencia Artificial (MULCIA)

AUTOR: Mariano Soto Fernández

================================================================================

DESCRIPCIÓN GENERAL
-------------------
Este proyecto implementa procesos de evaluación de embeddings generados a partir de
datos textuales con carga afectiva, extraídos de SenticNet como base de conocimiento
ontológica. Incluye tres escenarios de modelos (A, B y C), utilidades de gestión de
datos, métricas de evaluación y visualización de resultados.

--------------------------------------------------------------------------------
ESTRUCTURA DEL PROYECTO
--------------------------------------------------------------------------------

proyecto/
│
├── Code/                          # Código fuente principal
│   ├── config.py                  # Configuración global del proyecto (rutas, hiperparámetros,
│   │                              # constantes compartidas entre módulos)
│   ├── data_manager.py            # Gestión y preprocesamiento de datos: carga del dataset,
│   │                              # limpieza, tokenización y partición train/test
│   ├── main_modelA.ipynb          # Notebook del Modelo A: valuación de la primera aproximación
│   ├── main_modelB.ipynb          # Notebook del Modelo B: variante con nuevo encoder
│   │                              # 
│   ├── main_modelC.ipynb          # Notebook del Modelo C: variante con modelo del lenguaje
│   │                              # 
│   ├── metrics.py                 # Funciones de evaluación: distancia euclídea, similitud
│   │                              # coseno, coeficiente de silhouette, procesos de clustering y
|   |                              # pipeline de análisis
│   └── visuals.py                 # Generación de gráficos y visualizaciones: gráficos 3D y
│                                  # exploración inicial de los datos
│
└── datasets/                      # Datos y recursos ontológicos
    ├── license.txt                # Licencia de uso del dataset SenticNet
    └── senticnet_dataset.py       # Dataset SenticNet

--------------------------------------------------------------------------------
DESCRIPCIÓN DE COMPONENTES
--------------------------------------------------------------------------------

CONFIG (config.py)
  - Define rutas a los datasets.
  - Almacena hiperparámetros comunes (e.g., tamaño de batch, semillas).
  - Centraliza variables de entorno y flags de configuración.

DATA MANAGER (data_manager.py)
  - Carga y preprocesa los datos de entrada.
  - Integra información semántica de SenticNet en los ejemplos de entrenamiento.
  - Genera particiones reproducibles de los datos.

MODELOS (main_modelA/B/C.ipynb)
  - Cada notebook es autocontenido y ejecutable de forma independiente.
  - Sigue el flujo: carga de datos -> preparación de datos -> ejecución de pipeline 
    por caso de estudio.

MÉTRICAS (metrics.py)
  - Implementa funciones reutilizables de evaluación.
  - Permite comparar el rendimiento entre los tres escenarios de forma consistente.

VISUALIZACIONES (visuals.py)
  - Genera figuras para análisis exploratorio y reporte de resultados.
  - Las gráficas pueden exportarse como imágenes para su inclusión en informes
  (desde los cuadernos de Jupyter).

DATASET SENTICNET (datasets/)
  - senticnet_dataset.py expone una API de alto nivel para consultar conceptos,
    polaridades y dimensiones afectivas sin manipular el OWL directamente.

--------------------------------------------------------------------------------
REQUISITOS
--------------------------------------------------------------------------------

  - Python 3.13+
  - Jupyter Notebook / JupyterLab
  - Dependencias necesarias:
      pip install numpy pandas scikit-learn matplotlib plotly torch nltk
	sentence-transformers

--------------------------------------------------------------------------------
USO RÁPIDO
--------------------------------------------------------------------------------

  1. Configurar rutas y parámetros en Code/config.py.
  2. Verificar que se carga el conjunto de datos.
  3. Ejecutar los notebooks deseados:
       - main_modelA.ipynb
       - main_modelB.ipynb
       - main_modelC.ipynb
  4. Comparar resultados con las métricas generadas por metrics.py.

================================================================================
