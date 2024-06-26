import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from seaborn import scatterplot, kdeplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def nan_replace(tabel):
    for v in tabel.columns:
        if tabel[v].isna().any():
            if is_numeric_dtype(tabel[v]):
                tabel[v].fillna(tabel[v].mean(), inplace=True)
            else:
                tabel[v].fillna(tabel[v].mode()[0], inplace=True)

def plot_instante(tabel_z, coloana1, coloana2, y_test):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Plot instante axele " + coloana1 + " " + coloana2)
    scatterplot(tabel_z, x=coloana1, y = coloana2, hue=y_test, legend=True)

def plot_distributii(tabel_z, coloana, y_test):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Plot distributii pentru axa discriminanta " + coloana)
    kdeplot(tabel_z, x=coloana, hue=y_test, ax=ax, fill=True)

def plot_distributie(z, y, k=0):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Distributie in axa discriminanta " + str(k + 1), fontsize=16, color="m")
    kdeplot(x=z[:, k], hue=y, fill=True, ax=ax)

def evaluare_model(y_test, predictie, clase):

    c = confusion_matrix(y_test, predictie)

    acuratete = np.diag(c) * 100 / np.sum(c, axis=1)
    acuratete_medie = np.mean(acuratete)
    acuratete_globala = np.sum(np.diag(c)) * 100 / len(y_test)
    kappa = cohen_kappa_score(y_test, predictie)

    tabel_confuzie = pd.DataFrame(c, clase, clase)
    tabel_confuzie["Acuratete"] = acuratete

    tabel_indicatori = pd.Series([acuratete_medie, acuratete_globala, kappa],
                                 ["Acuratete medie", "Acuratete globala", "Index Cohen Kappa"],
                                 name="Indicatori")

    return tabel_confuzie, tabel_indicatori

def show():
    plt.show()