import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from seaborn import heatmap

def nan_replace(tabel):
    for v in tabel.columns:
        if tabel[v].isna().any().any():
            if is_numeric_dtype(tabel[v]):
                tabel[v].fillna(tabel[v].mean(), inplace=True)
            else:
                tabel[v].fillna(tabel[v].mode()[0], inplace=True)


def tabelare_varianta(alpha, etichete):
    return pd.DataFrame(data={
        "Varianta": alpha,
        "Varianta cumulata": np.cumsum(alpha),
        "Varianta procentuala": alpha*100/sum(alpha),
        "Varianta procentuala cumulata": np.cumsum(alpha) * 100 / sum(alpha)
    }, index=etichete)

def calcul_criterii(alpha, procent_minimal=80):
    m = len(alpha)

    #1. Criteriul variantei minime explicate
    k1 = np.where(np.cumsum(alpha) * 100/ sum(alpha) > procent_minimal)[0][0]

    #2. Criteriul Kaiser
    k2 = np.where(alpha < 1)[0][0] - 1

    #3. Criteriul Cattell
    eps = alpha[:(m-1)] - alpha[1:]
    sigma = eps[:(m-2)] - eps[1:]

    is_negativ = sigma < 0

    if any(is_negativ):
        k3 = np.where(is_negativ)[0][0] + 1
    else:
        k3 = None

    return (k1, k2, k3)


def plot_varianta(alpha, criterii):
    m = len(alpha)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Plot varianta")
    ax.set_xlabel("Cmponente")
    ax.set_ylabel("Varianta")
    x = np.arange(1, m+1)
    ax.set_xticks(x)
    ax.plot(x, alpha)
    ax.scatter(x, alpha)

    ax.axhline(alpha[criterii[0]], c="r", label="Criteriu varianta minima explicata")
    ax.axhline(alpha[criterii[1]], c="b", label="Criteriu Kaiser")
    ax.axhline(alpha[criterii[2]], c="g", label="Criteriul Cattell")

    ax.legend()

def corelograma(tabel, vmin=-1, titlu="Corelograma corelatii factoriale"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    heatmap(tabel, vmin=vmin, vmax=1, ax=ax, annot=True)

def cercul_corelatiilor(tabel, coloana1, coloana2):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Cercul corelatiilor")
    ax.set_xlabel(coloana1)
    ax.set_ylabel(coloana2)
    ax.axhline(0)
    ax.axvline(0)
    x = np.arange(0, np.pi*2, 0.01)
    ax.plot(np.cos(x), np.sin(x))
    ax.scatter(tabel[coloana1], tabel[coloana2],  color="r")

    for i in range(len(tabel)):
        ax.text(tabel[coloana1].iloc[i], tabel[coloana2].iloc[i], tabel.index[i])

def plot_scoruri(tabel, coloana1, coloana2):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Plot scoruri")
    ax.set_xlabel(coloana1)
    ax.set_ylabel(coloana2)
    ax.axhline(0)
    ax.axvline(0)
    ax.scatter(tabel[coloana1], tabel[coloana2])

    for  i in range(len(tabel)):
        ax.text(tabel[coloana1].iloc[i], tabel[coloana2].iloc[i], tabel.index[i])


def show():
    plt.show()


