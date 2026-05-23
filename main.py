


import importlib.util
import sys
from pathlib import Path
import torch

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer



PRINTLINE_SIZE = 60

NUMBERCOLS = [
    "introspection_value",
    "temper_value",
    "attitude_value",
    "sensitivity_value",
    "polarity_value"
]

CATCOLS = [
    "primary_emotion",
    "secondary_emotion",
    "polarity_label"
]

# ==============================================================
# Útiles
# ==============================================================

def show_data(df: pd.DataFrame):
    """Visualiza información del Pandas DataFrame"""

    # Datos muestra
    print("\n" + "="*PRINTLINE_SIZE)
    print(">>> Mostrando contenido del dataFrame:\n")
    print(df.head(5))
    print("...")
    print(df.tail(5))
    print("\n" + "="*PRINTLINE_SIZE)

    # Características: columnas, tamaño
    print("\n" + "="*PRINTLINE_SIZE)
    print("\n>>> nMostrando tamaño del dataFrame:\n")
    print(df.shape)
    print("\n>>> Mostrando atributos del dataFrame:\n")
    print(df.info())
    print("="*PRINTLINE_SIZE)

    # Variables numéricas
    ##
    # Estadísticas
    print("\n" + "="*PRINTLINE_SIZE)
    print("\n>>> Mostrando estadísticas de campos numéricos del dataFrame:\n")
    stats = df[NUMBERCOLS].agg([
        "mean", "median", "std", "var", "min", "max", "skew", "kurt"
    ])
    print(stats)

    # Valores nulos
    print("\n>>> Mostrando conteo de valores nulos por campo del dataFrame:\n")    
    nulls = df.isnull().sum()
    print(nulls)

    # porcentaje
    print("\n>>> Mostrando porcentaje de valores nulos (solo aquellos con valores nulos):\n")  
    nulls_pct = df.isnull().mean() * 100
    nulls_pct = nulls_pct[nulls_pct > 0]
    print(nulls_pct)
    print("\n" + "="*PRINTLINE_SIZE)

    # Correlaciones
    print("\n" + "="*PRINTLINE_SIZE)
    print("\n>>> Mostrando matriz de correlación:\n")  
    print(df[NUMBERCOLS].corr())
    print("="*PRINTLINE_SIZE)


    # Variables categóricas
    ##
    # Conteo de valores
    print("\n" + "="*PRINTLINE_SIZE)
    print("\n>>> Mostrando conteo categórico:")  
    for col in CATCOLS:
        print("\n")
        print(df[col].value_counts())
    print("="*PRINTLINE_SIZE)

    # Estadísticas agrupadas
    print("\n" + "="*PRINTLINE_SIZE)
    print("\n>>> Mostrando estadísticas agrupadas:") 
    # Media agrupada por emoción primaria
    print("\n\t [Media agrupada por Emoción primaria]")
    print("\n\t -------------------------------------\n")
    grouped = df.groupby("primary_emotion")[NUMBERCOLS].mean()
    print(grouped)

    # STD agrupada por emoción primaria
    print("\n\t [STD agrupada por Emoción primaria]")
    print("\n\t -----------------------------------\n")
    grouped_std = df.groupby("primary_emotion")[NUMBERCOLS].std()
    print(grouped_std)

    # Medias por polarización
    print("\n\t [Media agrupada por Polarización]")
    print("\n\t ---------------------------------\n")
    grouped_pol = df.groupby("polarity_label")[NUMBERCOLS].mean() 
    print(grouped_pol) 
    print("="*PRINTLINE_SIZE + "\n")


# ==============================================================
# Lectura del dataset
# ==============================================================

# --------------------------------------------------------------
# Conversión a DataFrame plano
# --------------------------------------------------------------

def read_dataset(datapath: str, columns: list = None) -> pd.DataFrame:
    """Importa senticnet_dataset.py y devuelve el diccionario senticnet."""
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

# ==============================================================
# Preprocesado
# ==============================================================

def clean_data(_df: pd.DataFrame) -> list:
    """
    Realiza el preprocesado del dataset:
        > Convierte las columnas de "primary_emotion" y "secondary_emotion"
          en dos codificaciones One-Hot.
        > Extrae las columnas "semantics" del conjunto
        > Elimina la columna "polarity_label"
        > Normaliza columnas numéricas
    """
    # Copia de seguridad
    df = _df.copy()


    # One Hot Encoding
    ##
    # Rellenar valores nulos para ohe
    df["secondary_emotion"] = df["secondary_emotion"].fillna("NONE")
    # Generar One Hot Encodings
    prim_ohe = pd.get_dummies(df["primary_emotion"]).astype(int)
    sec_ohe = pd.get_dummies(df["secondary_emotion"]).astype(int)
    # Eliminar características categóricas originales
    df = df.drop("primary_emotion", axis=1)
    df = df.drop("secondary_emotion", axis=1)
    # Unir ohes
    df = df.join(prim_ohe)
    df = df.join(sec_ohe, rsuffix='_seconday')
    df = df.drop("NONE", axis=1)


    # Crear DataFrame de semantics
    ##
    # Extraer semantics
    df_semantics = pd.DataFrame(df.get("semantics1"))
    df_semantics = df_semantics.join(df.get("semantics2"))
    df_semantics = df_semantics.join(df.get("semantics3"))
    df_semantics = df_semantics.join(df.get("semantics4"))
    df_semantics = df_semantics.join(df.get("semantics5"))
    # Eliminar semantics del df original
    df = df.drop("semantics1", axis=1)
    df = df.drop("semantics2", axis=1)
    df = df.drop("semantics3", axis=1)
    df = df.drop("semantics4", axis=1)
    df = df.drop("semantics5", axis=1)


    # Tratar categóricos restantes
    ##
    # Eliminar polarity_label (asumida por su valor)
    df = df.drop("polarity_label", axis=1)


    # Normalizar datos
    ##
    scaler = MinMaxScaler()
    df_numbercols_norm = pd.DataFrame(scaler.fit_transform(df[NUMBERCOLS]), columns=NUMBERCOLS, index=df.index)
    for c in NUMBERCOLS:
        df = df.drop(c, axis=1)
    df = df.join(df_numbercols_norm)

    return df, df_semantics



# ==============================================================
# Distancias (similitudes) reales
# ==============================================================




# ==============================================================
# Generación de embeddings
# ==============================================================

def gen_embeddings(df: pd.DataFrame):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)

    embeddings_texto = model.encode(df.index, batch_size=512, show_progress_bar=True)

    return embeddings_texto


# ==============================================================
# Distancias embeddings
# ==============================================================




# ==============================================================
# Comparativa - Correlación global


# Comparativa - Correlación local (top-k-neighbours)
# Jaccard


# precision@k


# recall@k


# ==============================================================




# ==============================================================
# Comparativa
# Comparativa - General


# Comparativa - Por emoción


# Comparativa - Por polaridad


# Comparativa - Semántica


# ==============================================================



# ==============================================================
# Flujo principal

def main():
    DATAPATH = "./datasets/senticnet_dataset.py"
    COLUMNS_NAMES = ['introspection_value', 'temper_value', 'attitude_value', 'sensitivity_value', 'primary_emotion', 'secondary_emotion', 'polarity_label', 'polarity_value', 'semantics1', 'semantics2', 'semantics3', 'semantics4', 'semantics5']
    data = read_dataset(DATAPATH, COLUMNS_NAMES)

    #show_data(data)
    df, df_semantics = clean_data(data)
# ==============================================================



if __name__ == "__main__":
    main()