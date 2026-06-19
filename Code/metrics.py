

# =============================================================================
# File:        metrics.py
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
#   Este archivo contiene todas las métricas que se utilizan para los procesos
#   de evaluación y análisis de este estudio
#
# Dependencies:
#   - config.py
#   - data_manager.py
#   - visuals.py
#
# =============================================================================


################################################################################
# IMPORTS
################################################################################

# =============================================================================
# Computación numérica y manipulación de datos
# =============================================================================
import numpy as np

# =============================================================================
# Machine Learning
# =============================================================================
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# =============================================================================
# Dependencias internas del proyecto
# =============================================================================
from config import (
    PRINTLINE_SIZE,
    RANDOM_SEED
)
from data_manager import gen_embeddings
from visuals import (
    plot_heatmap,
    plot_clusters_3d,
    plot_clusters_3d_plotly
)


################################################################################
# FUNCTIONS
################################################################################

# =============================================================================
# Similitud / Próximidad
# =============================================================================
def compute_projectedSemanticDistante(X: np.array, Y: np.array, mode="pairs", operation=euclidean_distances):
    """
    Calcula la similitud coseno o distancia euclidea entre datos

    Parameters
    ----------
    X : np.array
        Array de conceptos
    Y: np.array
        Array de conceptos o semánticos (según mode)
    mode: str
        Modo de operación:
        - "pairs": cómputo entre conceptos
        - "semantics": cómputo respecto semánticos
    operation:
        Operación a realizar:
        - euclidean_distances
        - cosine_similarity 

    Returns
    -------
    results: 
        Similitud/proximidad promedia
    """
    # Embeddings de X
    X_emb = gen_embeddings(X)
    results = []

    # Analisis de conjuntos
    if "pairs" == mode:
        Y_emb = gen_embeddings(Y)
        dist = operation(X_emb, Y_emb)
        return dist.mean()
    # Analisis de semantics
    elif "semantics" == mode:
    # Procesamiento por bloque
        for x, group in zip(X_emb, Y):
            # embeddings del grupo
            group_emb = gen_embeddings(np.array(group))
            dist = operation(x.reshape(1,-1), group_emb)
            results.append(np.mean(dist))
        return np.mean(results)
    else:
        raise Exception("ERROR - UNRECOGNISED INPUT MODE")
    

# =============================================================================
# Clustering
# =============================================================================
def calculate_clusters(X: np.array, emo_list: np.array, mode: str = "kmeans", metric: str="euclidean"):
    """
    Calcula el mejor cluster de los datos, con el coeficiente de silhouette como evaluador.

    Parameters
    ----------
    X : np.array
        Array de conceptos
    emo_list: np.array
        Conjunto de clases posibles (emociones)
    mode: str
        Modo de operación:
        - "kmeans": calcula mediante kmedias
        - "agglomerative": calcula mediante clustering aglomerativo
    metric: str
        Métrica a utilizar:
        - "euclidean"
        - "cosine"

    Returns
    -------
    results: dict ("best_score": float, "best_cluster": cluster, "target_score": float, "target_cluster": cluster)
        Diccionario con el mejor cluster y resultado, y los objetivos (resultado y cluster para k=num_classes)
    """

    results ={
        "best_score": -1,
        "best_cluster": None,
        "target_score": -1,
        "target_cluster": None
    }

    for k in range(2, len(emo_list)+1):
        if "kmeans" == mode:
            clusters_k = KMeans(k, init="k-means++", max_iter=500, random_state=RANDOM_SEED).fit(X)
        elif "agglomerative" == mode:
            linkage = "ward" if "euclidean" == metric else "average"
            clusters_k = AgglomerativeClustering(k, metric=metric, linkage=linkage).fit(X)
        else:
            raise Exception("ERROR - UNRECOGNISED CLUSTERING MODE")

        silk = silhouette_score(X, clusters_k.labels_, metric=metric)
        # Conservar el mejor
        if results["best_score"] <= silk:
            results["best_score"] = silk
            results["best_cluster"] = clusters_k
        # Guardar objetivo
        if k == len(emo_list):
            results["target_score"] = silk
            results["target_cluster"] = clusters_k

    return results


# =============================================================================
# Métricas conjuntas
# =============================================================================

# Análsis de semánticos y coherencia interna
def compute_allprojectedSemanticDistance(X: np.array, Xs: np.array):
    """
    Calcula las similitudes y distanticas, tanto internas de X como con sus semánticos

    Parameters
    ----------
    X : np.array
        Array de conceptos
    Xs: np.array
        Array de semánticos

    Returns
    -------
    eucD_global:
        Distrancia euclidea promedia entre conceptos de X
    cosS_global: np.array
        Similitud coseno promedia entre conceptos de X
    ucD_sem:
        Distrancia euclidea promedia entre X y ssus semánticos
    cosS_sem: np.array
        Similitud coseno promedia entre X y ssus semánticos
    """
    eucD_global = compute_projectedSemanticDistante(X, X, mode="pairs", operation=euclidean_distances)
    cosS_global = compute_projectedSemanticDistante(X, X, mode="pairs", operation=cosine_similarity)
    eucD_sem = compute_projectedSemanticDistante(X, Xs, mode="semantics", operation=euclidean_distances)
    cosS_sem = compute_projectedSemanticDistante(X, Xs, mode="semantics", operation=cosine_similarity)

    return eucD_global, cosS_global, eucD_sem, cosS_sem


# Análisis de similitud por emoción
def emoVSemo(X: np.array, y: np.array):
    """
    Calcula las similitudes y distanticas, entre conjuntos de emociones

    Parameters
    ----------
    X : np.array
        Array de conceptos
    y: np.array
        Objetivos

    Returns
    -------
    results_euc: np.ndarray
        Matriz de distancias
    results_cos: np.ndarray
        Matriz de similitudes
    """

    emo_list = np.unique(y)
    X = np.array(X)
    y = np.array(y).ravel()

    results_euc = np.zeros((len(emo_list), len(emo_list)))
    results_cos = np.zeros((len(emo_list), len(emo_list)))

    for i, emo1 in enumerate(emo_list):
        for j, emo2 in enumerate(emo_list):
            # Embedding
            X_emo1 = X[y == emo1]
            X_emo2 = X[y == emo2]
            results_euc[i][j] = euclidean_distances(X_emo1, X_emo2).mean()
            results_cos[i][j] = cosine_similarity(X_emo1, X_emo2).mean()
    
    return results_euc, results_cos


# Análisis por clusterización
def calculate_allClusters(X, X_emb, classes):
    """
    Calcula los mejores clusteres de los datos

    Parameters
    ----------
    X : np.array
        Array de conceptos
    X_emb: np.array
        Array de embeddings
    classes:
        Conjunto de objetivos

    Returns
    -------
    res_rk: dict ("best_score": float, "best_cluster": cluster)
        Diccionario con el mejor cluster y resultado de los datos reales
        mediante kmeans
    res_rae: dict ("best_score": float, "best_cluster": cluster)
        Diccionario con el mejor cluster y resultado de los datos reales
        mediante aglomerativo con distancia euclídea
    res_rac: dict ("best_score": float, "best_cluster": cluster)
        Diccionario con el mejor cluster y resultado de los datos reales
        mediante aglomerativo con similitud coseno
    res_ek: dict ("best_score": float, "best_cluster": cluster)
        Diccionario con el mejor cluster y resultado de los embeddings
        mediante kmeans
    res_eae: dict ("best_score": float, "best_cluster": cluster)
        Diccionario con el mejor cluster y resultado de los embeddings
        mediante aglomerativo con distancia euclídea
    res_eac dict ("best_score": float, "best_cluster": cluster)
        Diccionario con el mejor cluster y resultado de los embeddings
        mediante aglomerativo con similitud coseno
    """

    # Real
    res_rk = calculate_clusters(X, classes, mode="kmeans")
    res_rae = calculate_clusters(X, classes, mode="agglomerative", metric="euclidean")
    res_rac = calculate_clusters(X, classes, mode="agglomerative", metric="cosine")
    # Proyectado
    res_ek = calculate_clusters(X_emb, classes, mode="kmeans")
    res_eae = calculate_clusters(X_emb, classes, mode="agglomerative", metric="euclidean")
    res_eac = calculate_clusters(X_emb, classes, mode="agglomerative", metric="cosine")

    return res_rk, res_rae, res_rac, res_ek, res_eae, res_eac


# Análisis completo
def run_allMetrics(Xt, Xn, Xs, Xe, y, figsize=(8,8)):
    """
    Calcula todas las métricas para el estudio de los datos e imprime los resultados

    Parameters
    ----------
    Xt: np.array
        Array de conceptos textuales
    Xn: np.array
        Array de conceptos cuantificados
    Xs: np.array
        Array de semánticos
    Xe: np.array
        Array de embeddings
    y: np.array
        Objetivos
    figsize:
        Tamaño de las imágenes

    Returns
    -------

    """
    emo_list = np.unique(y)

    # Cohesión interna - global y por semánticos
    ##
    print("="*PRINTLINE_SIZE)
    print("Cálculo de la cohesión interna y similitud con semánticos")
    print("="*PRINTLINE_SIZE)

    eucD_global, cosS_global, eucD_sem, cosS_sem = compute_allprojectedSemanticDistance(Xt, Xs)

    print(f"Distancia euclidea global:\t\t {eucD_global}")
    print(f"Similitud coseno global:\t\t {cosS_global}")
    print(f"Distancia euclidea por semánticos:\t {eucD_sem}")
    print(f"Similitud coseno por semánticos:\t {cosS_sem}")

    # Cohesión por emoción
    ##
    print("="*PRINTLINE_SIZE)
    print("Cálculo de la similitud entre conceptos por emoción")
    print("="*PRINTLINE_SIZE)

    hm_euc_eve, hm_cos_eve = emoVSemo(Xe, y)

    plot_heatmap(hm_euc_eve, emo_list, title="Distancias euclídeas", figsize=figsize)
    plot_heatmap(hm_cos_eve, emo_list, title="Similitudes coseno", figsize=figsize)

    # Clustering
    ##
    print("="*PRINTLINE_SIZE)
    print("Cálculo de los mejores clusters")
    print("="*PRINTLINE_SIZE)

    res_rk, res_rae, res_rac, res_ek, res_eae, res_eac = calculate_allClusters(Xn, Xe, emo_list)
    
    # Mejores coeficiente de Silhouette
    print(f"Coef. Silhouette - Real KMeans       [k={res_rk["best_cluster"].n_clusters}]: {res_rk["best_score"]}")
    print(f"Coef. Silhouette - Real Agglom (Euc) [k={res_rae["best_cluster"].n_clusters}]: {res_rae["best_score"]}")
    print(f"Coef. Silhouette - Real Agglom (Cos) [k={res_rac["best_cluster"].n_clusters}]: {res_rac["best_score"]}")
    print(f"Coef. Silhouette - Embe KMeans       [k={res_ek["best_cluster"].n_clusters}]: {res_ek["best_score"]}")
    print(f"Coef. Silhouette - Embe Agglom (Euc) [k={res_eae["best_cluster"].n_clusters}]: {res_eae["best_score"]}")
    print(f"Coef. Silhouette - Embe Agglom (Cos) [k={res_eac["best_cluster"].n_clusters}]: {res_eac["best_score"]}")
    # Mejores clusters - Gráficas fijas
    plot_clusters_3d(Xn, res_rk["best_cluster"].labels_, title=f"Real-kmeans (k={res_rk["best_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xn, res_rae["best_cluster"].labels_, title=f"Real-agglom euc. (k={res_rae["best_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xn, res_rac["best_cluster"].labels_, title=f"Real-agglom cos. (k={res_rac["best_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xe, res_ek["best_cluster"].labels_, title=f"Embe-kmeans (k={res_ek["best_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xe, res_eae["best_cluster"].labels_, title=f"Embe-agglom euc. (k={res_eae["best_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xe, res_eac["best_cluster"].labels_, title=f"Embe-agglom cos. (k={res_eac["best_cluster"].n_clusters})", figsize=figsize)
    #Mejores clusters - Gráficas dinámicas
    plot_clusters_3d_plotly(Xn, res_rk["best_cluster"].labels_,  y, title=f"Real-kmeans (k={res_rk["best_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xn, res_rae["best_cluster"].labels_, y, title=f"Real-agglom euc. (k={res_rae["best_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xn, res_rac["best_cluster"].labels_, y, title=f"Real-agglom cos. (k={res_rac["best_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xe, res_ek["best_cluster"].labels_,  y, title=f"Embe-kmeans (k={res_ek["best_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xe, res_eae["best_cluster"].labels_, y, title=f"Embe-agglom euc. (k={res_eae["best_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xe, res_eac["best_cluster"].labels_, y, title=f"Embe-agglom cos. (k={res_eac["best_cluster"].n_clusters})")

    print("="*PRINTLINE_SIZE)
    print("Cálculo de los clusters para k=Número_emociones")
    print("="*PRINTLINE_SIZE)
    
    # Mejores coeficiente de Silhouette
    print(f"Coef. Silhouette - Real KMeans       [k={res_rk["target_cluster"].n_clusters}]: {res_rk["target_score"]}")
    print(f"Coef. Silhouette - Real Agglom (Euc) [k={res_rae["target_cluster"].n_clusters}]: {res_rae["target_score"]}")
    print(f"Coef. Silhouette - Real Agglom (Cos) [k={res_rac["target_cluster"].n_clusters}]: {res_rac["target_score"]}")
    print(f"Coef. Silhouette - Embe KMeans       [k={res_ek["target_cluster"].n_clusters}]: {res_ek["target_score"]}")
    print(f"Coef. Silhouette - Embe Agglom (Euc) [k={res_eae["target_cluster"].n_clusters}]: {res_eae["target_score"]}")
    print(f"Coef. Silhouette - Embe Agglom (Cos) [k={res_eac["target_cluster"].n_clusters}]: {res_eac["target_score"]}")
    # Mejores clusters - Gráficas fijas
    plot_clusters_3d(Xn, res_rk["target_cluster"].labels_, title=f"Real-kmeans (k={res_rk["target_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xn, res_rae["target_cluster"].labels_, title=f"Real-agglom euc. (k={res_rae["target_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xn, res_rac["target_cluster"].labels_, title=f"Real-agglom cos. (k={res_rac["target_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xe, res_ek["target_cluster"].labels_, title=f"Embe-kmeans (k={res_ek["target_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xe, res_eae["target_cluster"].labels_, title=f"Embe-agglom euc. (k={res_eae["target_cluster"].n_clusters})", figsize=figsize)
    plot_clusters_3d(Xe, res_eac["target_cluster"].labels_, title=f"Embe-agglom cos. (k={res_eac["target_cluster"].n_clusters})", figsize=figsize)
    #Mejores clusters - Gráficas dinámicas
    plot_clusters_3d_plotly(Xn, res_rk["target_cluster"].labels_,  y, title=f"Real-kmeans (k={res_rk["target_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xn, res_rae["target_cluster"].labels_, y, title=f"Real-agglom euc. (k={res_rae["target_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xn, res_rac["target_cluster"].labels_, y, title=f"Real-agglom cos. (k={res_rac["target_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xe, res_ek["target_cluster"].labels_,  y, title=f"Embe-kmeans (k={res_ek["target_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xe, res_eae["target_cluster"].labels_, y, title=f"Embe-agglom euc. (k={res_eae["target_cluster"].n_clusters})")
    plot_clusters_3d_plotly(Xe, res_eac["target_cluster"].labels_, y, title=f"Embe-agglom cos. (k={res_eac["target_cluster"].n_clusters})")


