

# =============================================================================
# File:        data_manager.py
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
#   Este archivo contiene todas las funciones para la gestión y procesado de
#   los datos con lo que se trabaja en este estudio (base de datos SenticNet)
#
# Dependencies:
#   - config.py
#
# =============================================================================


################################################################################
# IMPORTS
################################################################################

# =============================================================================
# Librerías estándar de Python
# =============================================================================
import sys
from pathlib import Path

# =============================================================================
# Computación numérica y manipulación de datos
# =============================================================================
import importlib.util
import numpy as np
import pandas as pd

# =============================================================================
# Machine Learning
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.corpus import wordnet as wn


# =============================================================================
# Dependencias internas del proyecto
# =============================================================================
from config import (
    STEMMER,
    LEMMATIZER,
    ENCODER_A,
    ENCODER_BATCH_SIZE,
    RANDOM_SEED,
    NUMERIC_COLUMNS
)


################################################################################
# FUNCTIONS
################################################################################

# =============================================================================
# Lectura de datos
# =============================================================================
def read_dataset(datapath: str, columns: list = None) -> pd.DataFrame:
    """
    Importa senticnet_dataset.py y devuelve el diccionario senticnet.

    Parameters
    ----------
    datapath: str
        Ruta al dataset.
    columns: list
        Conjunto de columna / atributos del dataset.

    Returns
    -------
    datos: pd.DataFrame
        Dataset en forma de DataFrame de Pandas.
    """

    datapath = Path(datapath)
    if not datapath.exists():
        sys.exit(f"[M1] ERROR: no se encuentra el fichero '{datapath}'.\n")

    print(f"[M1] Cargando {datapath} ...")
    spec = importlib.util.spec_from_file_location("senticnet_dataset", datapath)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    if not hasattr(modulo, "senticnet"):
        sys.exit("[M1] ERROR: el módulo no contiene una variable llamada 'senticnet'.")

    datos = pd.DataFrame.from_dict(modulo.senticnet, orient="index", columns=columns)
    print(f"[M1] Registros cargados: {len(datos):,}")
    return datos


# =============================================================================
# Preprocesamineto
# =============================================================================
# Normalizado de textos (stemming y lematizing)
def normalize_texts(df: pd.DataFrame, y: np.array, semantics: dict, transform, on_conflict: str="first"):
    """
    Normaliza los textos del índice y de 'semantics', y elimina registros
    duplicados (mismo texto tras normalizar) teniendo en cuenta la
    clasificación asociada a cada fila (correspondiente en 'y')).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame indexado por texto.
    y: array-like
        Clasificación de cada fila de 'df' (mismo orden y longitud).
    semantics: dict
        Diccionario {clave: [valores]} a normalizar igual que el índice.
    transform: callable
        Función de normalización de texto. Por defecto, identidad.
    on_conflict: {"first", "drop", "raise"}
        Qué hacer cuando dos registros normalizan al mismo texto
        pero tienen clasificaciones distintas:
          - "first": se queda con la primera ocurrencia (ignora el conflicto).
          - "drop": elimina todo el grupo en conflicto.
          - "raise": lanza ValueError para revisión manual.

    Returns
    -------
    df: pd.DataFrame
    labels: np.ndarray
        Alineado con las filas resultantes de 'df'.
    semantics: dict
    """
    if transform is None:
        transform = lambda x: x

    labels = y.ravel()

    if len(labels) != len(df):
        raise ValueError("labels debe tener la misma longitud que df")
    
    df = df.copy()
    df.insert(0, "y", labels)
    df.index = [transform(x) for x in df.index]

    df = df[~df.index.duplicated(keep=on_conflict)]
    labels = df["y"].to_numpy()
    df = df.drop(columns=["y"])

    semantics = {
        transform(k): [transform(x) for x in v]
        for k, v in semantics.items()
    }
    
    return df, labels, semantics


# Pipeline de preprocesado
def preprocess_data(_df: pd.DataFrame, stemming: bool=False, lemmatize: bool=False) -> list:
    """
    Realiza el preprocesado del dataset:
        > Extrae las columnas de "primary_emotion" y "secondary_emotion"
          para usarlas de discriminantes.
        > Extrae las columnas "semantics" del conjunto
        > Elimina la columna "polarity_label"
        > Normaliza columnas numéricas

    Parameters
    ----------
    _df: pd.DataFrame
        Conjunto de datos
    stemming: bool
        Flag que indica si se hace o no stemming
    lemmatize: bool
        Flag que indica si se hace o no lemmatize

    Returns
    -------
    df: pd.DataFrame
        Conjunto de datos original preprocesado
    df_stem: pd.DataFrame
        Conjunto de datos preprocesado + stemming
    df_lemma: pd.DataFrame
        Conjunto de datos preprocesado + lemmatize
    y: np.array
        Objetivo acorde a df
    y_stem: np.array
        Objetivo acorde a df_stem
    y_lemma: np.array
        Objetivo acorde a df_lemma
    semantics: dict (["concepto"]: np.array("semánticos"))
        Semánticos acordes a df
    sem_stem: dict (["concepto"]: np.array("semánticos"))
        Semánticos acordes a df_stem y preprocesados con stemming
    sem_lemma: dict (["concepto"]: np.array("semánticos"))
        Semánticos acordes a df_lemma y preprocesados con lemmatize
    """
    # Copia de seguridad
    _df = _df.copy()
    df = _df.drop_duplicates(subset=NUMERIC_COLUMNS)
    df = df.dropna()

    # Crear DataFrame de emotions
    ##
    df_emotions =  pd.DataFrame(df.get("primary_emotion"))
    # df_emotions = df_emotions.join(df.get("secondary_emotion"))

    # Crear DataFrame de semantics
    ##
    df_semantics = df[
        ["semantics1", "semantics2", "semantics3", "semantics4", "semantics5"]
    ]

    # Eliminar categóricos
    ##
    # Eliminar polarity_label (asumida por su valor)
    df = df.drop(["polarity_label"], axis=1)

    # Eliminar "emotions" (extraídas a parte)
    df = df.drop(["primary_emotion", "secondary_emotion"], axis=1)

    # Eliminar "semantics" (extraídas a parte)
    df = df.drop(["semantics1","semantics2","semantics3","semantics4","semantics5"], axis=1)

    # Normalizar datos
    ##
    df = pd.DataFrame(StandardScaler().fit_transform(df[NUMERIC_COLUMNS]), columns=NUMERIC_COLUMNS, index=df.index)

    y = df_emotions.to_numpy()
    semantics = dict(zip(df_semantics.index, df_semantics.to_numpy()))
    if stemming:
      df_stem, y_stem, sem_stem = normalize_texts(df, y, semantics, transform=STEMMER)
    if lemmatize:
      df_lemma, y_lemma, sem_lemma = normalize_texts(df, y, semantics, transform=LEMMATIZER)
      
    return df, df_stem, df_lemma, y, y_stem, y_lemma, semantics, sem_stem, sem_lemma


# Particionar datos
def get_splits(X: np.array, Y: np.array, df_data: pd.DataFrame, sem: dict, encoder=ENCODER_A, p:float = 0.5):
    """
    Divide el conjunto de datos y prepara las particiones asociadas.

    Parameters
    ----------
    X : np.array
        Array Conceptos
    Y : np.array
        Objetivos acorde a X
    df_data: pd.DataFrame
        DataFrame con cuantificación de los conceptos
    sem: dict
        Conjuntos de semánticos acorde a X
    encoder:
        Modelo a utilizar para generar los embeddings
    p: float
        Tamaño del subconjunto de datos a extraer.
        p es porcentaje si p<1 y será cantidad si p >= 1

    Returns
    -------
    Xt: np.array
        Array de conceptos textuales
    Xn: np.array
        Array de conceptos cuantificados
    Xs: np.array
        Array de semánticos por conceptos
    Xe: np.array
        Array de embeddings de concepto
    y:  np.array
        Objetivos
    """
        
    if p < 1:
        test_size = (1-p) 
    else:
        test_size = len(X)-p
    Xt, _, y, _ = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=RANDOM_SEED)
    # Preparar numéricos
    Xn = df_data.loc[Xt]
    # Preparar semánticos
    Xs = np.array([sem[k] for k in Xt])
    # Generar embeddings
    Xe = gen_embeddings(Xt, encoder=encoder)

    return Xt, Xn, Xs, Xe, y


# Extraer definición de concepto
def get_concept_text(concept: str):
    """
    Extrae una definición del concepto de WordNet

    Parameters
    ----------
    concept: str
        Concepto a definir

    Returns
    -------
    str
        Cadena con el concepto enriquecido por un ejemplo de uso si
        lo tiene; si no, por su definición, si se ha encontrado una; si no, se devuelve el propio
        concepto.
    """ 
    # Caso 1. Lookup directo
    synsets = wn.synsets(concept)
    if synsets:
        if len(synsets[0].examples()) == 1:
            return f"{synsets[0].examples()}"
        elif len(synsets[0].examples()) > 1:
            return f"{synsets[0].examples()[0]}"
        return f"{synsets[0].definition()}"
    
    # Caso 2. Con espacios
    term = concept.replace("_", " ")
    synsets = wn.synsets(term)
    if synsets:
        if len(synsets[0].examples()) == 1:
            return f"{synsets[0].examples()}"
        elif len(synsets[0].examples()) > 1:
            return f"{synsets[0].examples()[0]}"
        return f"{synsets[0].definition()}"
    
    # Caso 3. Palabra más larga del concepto
    words = term.split()
    head = max(words, key=lambda w: len(wn.synsets(w)))
    synsets = wn.synsets(head)
    if synsets:
        if len(synsets[0].examples()) == 1:
            return f"{synsets[0].examples()}"
        elif len(synsets[0].examples()) > 1:
            return f"{synsets[0].examples()[0]}"
        return f"{synsets[0].definition()}"
    
    # Caso 4. Default: concept
    return concept


# Generar verbalización
def gen_verbalization(X: np.array, sem: dict, mode: str="keep"):
    """
    Genera una verbalización de los datos.

    Parameters
    ----------
    X : np.array
        Array de conceptos
    sem: dict ("concept": semánticos)
        Dictionarios de semánticos por concepto
    mode: str
        Tratamiento de los conceptos no presentes en WordNet:
        - "keep": conserva el concepto como su propia definición
        - "drop": elimina el concepto del conjunto resultante


    Returns
    -------
    Xv: np.array
        Array de conceptos verbalizados
    Xvs: np.array
        Array de semánticos verbalizados por concepto
    mask: np.array
        Máscara de instancias conservadas
    """
    Xv = []
    Xvs = []
    mask = []
    for x in X:
        # Verbalizar conceptos
        gloss = get_concept_text(x)
        if x == gloss and "drop" == mode:
            mask.append(False)
            continue

        mask.append(True)
        Xv.append(f"{x}: {gloss}")

        # Verbalizar semánticos
        vs = []
        for s in sem[x]:
            gloss = get_concept_text(s)
            vs.append(f"{s}: {gloss}")

        Xvs.append(np.array(vs))
    Xv = np.array(Xv)
    mask = np.array(mask)
    return Xv, Xvs, mask


# =============================================================================
# Generación de embeddings
# =============================================================================
def gen_embeddings(X: np.array, encoder=ENCODER_A):
    """
    Genera los embeddings de X

    Parameters
    ----------
    X : np.array
        Array de conceptos
    encoder:
        Modelo a utilizar para generar los embeddings
    Returns
    -------
    embeddings_texto:
        Array de embeddings
    """
    embeddings_texto = encoder.encode(X, batch_size=ENCODER_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
    return embeddings_texto


