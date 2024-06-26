import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from seaborn import heatmap
import matplotlib.pyplot as plt


def nan_replace(tabel):
    for v in tabel:
        if tabel[v].isna().any():
            if is_numeric_dtype(tabel[v]):
                tabel[v].fillna(tabel[v].mean(), inplace=True)
            else:
                tabel[v].fillna(tabel[v].mode()[0], inplace=True)

def tabelare_varianta(alpha):
    return pd.DataFrame(data={
        "Varianta": alpha[0],
        "Varianta cumulata": np.cumsum(alpha[0]),
        "Varianta procentuala": alpha[1]*100,
        "Varianta procentuala cumulata": alpha[2]*100
    }
    )

def corelograma(tabel, vmin=-1, titlu="Corelograma corelatii factoriale"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    heatmap(tabel, vmin=vmin, vmax=1, ax=ax, annot=True)

def plot_corelatii(tabel, coloana1, coloana2, titlu="Cercul corelatiilor"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    ax.set_xlabel(coloana1)
    ax.set_ylabel(coloana2)
    ax.axhline(0)
    ax.axvline(0)
    x = np.arange(0, np.pi*2, 0.01)
    ax.plot(np.cos(x), np.sin(x))
    ax.scatter(tabel[coloana1], tabel[coloana2])

    for i in range(len(tabel)):
        ax.text(tabel[coloana1].iloc[i], tabel[coloana2].iloc[i], tabel.index[i])

def plot_scoruri(tabel, coloana1, coloana2, titlu="Plot scoruri"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    ax.set_xlabel(coloana1)
    ax.set_ylabel(coloana2)
    ax.axhline(0)
    ax.axvline(0)
    ax.scatter(tabel[coloana1], tabel[coloana2])

    for i in range(len(tabel)):
        ax.text(tabel[coloana1].iloc[i], tabel[coloana2].iloc[i], tabel.index[i])


def show():
    plt.show()