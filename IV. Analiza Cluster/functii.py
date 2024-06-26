import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_silhouette
from seaborn import scatterplot


def nan_replace(tabel):
    for v in tabel.columns:
        if tabel[v].isna().any():
            tabel[v].fillna(tabel[v].mean(), inplace=True)
        else:
            tabel[v].fillna(tabel[v].mode()[0], inplace=True)


def calcul_partitie(h, nr_clusteri=None):
    m = np.shape(h)[0]
    n = m+1

    if nr_clusteri is None:
        dif = h[1:, 2] - h[:(m-1), 2]
        j = np.argmax(dif)
        nr_clusteri = m - j
    else:
        j = m - nr_clusteri

    threshold = (h[j, 2] + h[j+1, 2]) / 2

    c = np.arange(n)

    for i in range(m - nr_clusteri +1):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c==k1] = n+i
        c[c==k2] = n+i

    codes = pd.Categorical(c).codes

    partitie = np.array(["C" + str(i+1) for i in codes])

    return nr_clusteri, threshold, partitie

def dendograma(h, threshold, etichete):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Dendograma")
    dendrogram(h, color_threshold=threshold, ax=ax, labels=etichete)


def plot_silh(x, partitie):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Plot indecsi sihlouette")
    plot_silhouette(x, partitie, ax=ax)

def histograma(set_date, variabila, partitie):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Histograma variabila " + variabila)
    clase = np.unique(partitie)
    q = len(clase)
    min_max = (set_date[variabila].min(), set_date[variabila].max())
    ax = fig.subplots(1, q, sharey=True)
    for i in range(q):
        axe = ax[i]
        axe.set_xlabel(str(clase[i]))
        axe.hist(set_date[partitie == clase[i]][variabila], rwidth=0.9)

def plot_axe_principale(tabel_z, coloana1, coloana2, partitie):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Partitie in axele Z1 si Z2")
    scatterplot(tabel_z, x=coloana1, y=coloana2, ax=ax, hue=partitie)

    for i in range(len(tabel_z)):
        ax.text(tabel_z[coloana1].iloc[i], tabel_z[coloana2].iloc[i], tabel_z.index[i])




def show():
    plt.show()