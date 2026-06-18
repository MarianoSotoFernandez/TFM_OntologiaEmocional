

# =============================================================================
# File:        visuals.py
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
#   Este archivo contiene todas las funciones para la representación de datos
#   que se utilizan en este estudio.
#
# Dependencies:
#   - config.py
#
# =============================================================================


################################################################################
# IMPORTS
################################################################################

# =============================================================================
# Computación numérica y manipulación de datos
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# Machine Learning
# =============================================================================
from sklearn.manifold import TSNE

# =============================================================================
# Visualización
# =============================================================================
import matplotlib.pyplot as plt
import plotly.express as px

# =============================================================================
# Dependencias internas del proyecto
# =============================================================================
from config import (
    PRINTLINE_SIZE,
    RANDOM_SEED,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS
)


################################################################################
# FUNCTIONS
################################################################################

# =============================================================================
#   Funciones de visualización
# =============================================================================

# Datos crudos
def show_raw_data(df: pd.DataFrame):
    """
    Visualiza información del Pandas DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        Dataset

    Returns
    -------

    """

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
    stats = df[NUMERIC_COLUMNS].agg([
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
    print(df[NUMERIC_COLUMNS].corr())
    print("="*PRINTLINE_SIZE)


    # Variables categóricas
    ##
    # Conteo de valores
    print("\n" + "="*PRINTLINE_SIZE)
    print("\n>>> Mostrando conteo categórico:")  
    for col in CATEGORICAL_COLUMNS:
        print("\n")
        print(df[col].value_counts())
    print("="*PRINTLINE_SIZE)

    # Estadísticas agrupadas
    print("\n" + "="*PRINTLINE_SIZE)
    print("\n>>> Mostrando estadísticas agrupadas:") 
    # Media agrupada por emoción primaria
    print("\n\t [Media agrupada por Emoción primaria]")
    print("\n\t -------------------------------------\n")
    grouped = df.groupby("primary_emotion")[NUMERIC_COLUMNS].mean()
    print(grouped)

    # STD agrupada por emoción primaria
    print("\n\t [STD agrupada por Emoción primaria]")
    print("\n\t -----------------------------------\n")
    grouped_std = df.groupby("primary_emotion")[NUMERIC_COLUMNS].std()
    print(grouped_std)

    # Medias por polarización
    print("\n\t [Media agrupada por Polarización]")
    print("\n\t ---------------------------------\n")
    grouped_pol = df.groupby("polarity_label")[NUMERIC_COLUMNS].mean() 
    print(grouped_pol) 
    print("="*PRINTLINE_SIZE + "\n")


# Heatmap
def plot_heatmap(X: np.ndarray, labels: np.array=None, title:str="Heatmap", figsize=(8, 8)):
    """
    Genera un heatmap de los datos

    Parameters
    ----------
    X : np.array
        Array de valores para el heatmap
    labels: np.array
        Conjunto de etiquetas únicas
    title: str
        Título del imágen
    figsize:
        Tamaño de la imágen

    Returns
    -------

    """

    plt.figure(figsize=figsize)
    im = plt.imshow(X, aspect="auto")

    plt.title(title)
    plt.colorbar(im)

    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)

    plt.tight_layout()
    plt.show()

    
# Static Clusters
def plot_clusters_3d(X, labels,
    title=None,
    figsize=(8, 8),
    perplexity=30,
    random_state=RANDOM_SEED,
    cmap='tab20',
    s=10
):
    """
    Genera un gráfico 3D de los datos

    Parameters
    ----------
    X : np.array
        Array de valores para el heatmap
    labels: np.array
        Conjunto de etiquetas de los clusters
    title: str
        Título del imágen
    figsize:
        Tamaño de la imágen

    Returns
    -------

    """
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=random_state
    )

    X_3d = tsne.fit_transform(X)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        X_3d[:, 0],
        X_3d[:, 1],
        X_3d[:, 2],
        c=labels,
        cmap=cmap,
        s=s
    )

    if title:
        ax.set_title(title)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    plt.show()

    return fig, ax


# Dinamic Clusters
def plot_clusters_3d_plotly(X: np.array, cluster_labels: np.array, true_labels: np.array=None,
    title= "Cluster",
    perplexity=30,
    random_state=RANDOM_SEED,
):
    """
    Genera un heatmap de los datos

    Parameters
    ----------
    X : np.array
        Array de valores para el heatmap
    labels: np.array
        Conjunto de etiquetas de los clusters
    true_labels: np.array
        Objetivos de X
    title: str
        Título del imágen

    Returns
    -------

    """
    
    true_labels = true_labels.ravel()
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=random_state
    )

    X_3d = tsne.fit_transform(X)

    df = pd.DataFrame({
        "x": X_3d[:, 0],
        "y": X_3d[:, 1],
        "z": X_3d[:, 2],
        "cluster": cluster_labels.astype(str)
    })

    if true_labels is not None:
        df["class"] = true_labels.astype(str)

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="class",
        symbol="cluster",
        opacity=0.8,
        title=title
    )

    fig.update_traces(marker=dict(size=4))

    fig.update_layout(
        legend_title_text="Clusters",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


