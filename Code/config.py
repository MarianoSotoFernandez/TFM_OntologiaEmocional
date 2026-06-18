

# =============================================================================
# File:        config.py
#
# Author:      Mariano Soto Fernández
#
# Institution: Escuela Técnica Superior de Ingeniería Informática,
#              Universidad de Sevilla
#
# Degree:      Máster en Computación, Lógica e Inteligencia Artificial (MULCIA)
#
# Academic year: 2025-2026
#
#
# Version:     1.0
#
# Description:
#   Este archivo contiene la configuración (variables globales) de todo el
#   proyecto
#
# Dependencies:
#
#
# =============================================================================


################################################################################
# IMPORTS
################################################################################

# =============================================================================
# Procesamiento del lenguaje natural (NLP)
# =============================================================================
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# =============================================================================
# Deep Learning
# =============================================================================
import torch


################################################################################
# GLOBAL CONFIGURATION
################################################################################

# =============================================================================
# NLP
# =============================================================================

STEMMER = lambda x: PorterStemmer().stem(x)
LEMMATIZER = lambda x: WordNetLemmatizer().lemmatize(x)

# =============================================================================
# Modelo de embeddings
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2",
    device=DEVICE
)

ENCODER_BATCH_SIZE = 512

# =============================================================================
# Reproducibilidad
# =============================================================================

RANDOM_SEED = 54

# =============================================================================
# Visualización
# =============================================================================

PRINTLINE_SIZE = 60

# =============================================================================
# Rutas de datos
# =============================================================================

DATAPATH = "../datasets/senticnet_dataset.py"

# =============================================================================
# Definición de las columnas del dataset
# =============================================================================

# Columnas numéricas
NUMERIC_COLUMNS = [
    "introspection_value",
    "temper_value",
    "attitude_value",
    "sensitivity_value",
    "polarity_value",
]

# Columnas categóricas
CATEGORICAL_COLUMNS = [
    "primary_emotion",
    "secondary_emotion",
    "polarity_label",
]

# Columnas del dataset
COLUMN_NAMES = [
    "introspection_value",
    "temper_value",
    "attitude_value",
    "sensitivity_value",
    "primary_emotion",
    "secondary_emotion",
    "polarity_label",
    "polarity_value",
    "semantics1",
    "semantics2",
    "semantics3",
    "semantics4",
    "semantics5",
]


