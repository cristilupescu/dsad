import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2
import matplotlib.pyplot as plt
from seaborn import heatmap


def nan_replace_t(t):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if any(t[v].isna()):
            if is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(), inplace=True)
            else:
                t[v].fillna(t[v].mode()[0], inplace=True)


def test_bartlett(r2, n, p, q, m):
    x = 1 - r2
    df = [(p - k + 1) * (q - k + 1) for k in range(1, m+1)]
    l = np.flip(np.cumprod(np.flip(x)))
    chi2_ = (- n + 1 + (p + q + 1) /2) * np.log(l)
    return 1- chi2.cdf(chi2_, df)

def cercul_corelatiilor(tabel_r_xz, z1, z2, tabel_r_yu, u1, u2):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(z1 + "/" + u1)
    ax.set_ylabel(z2 + "/" + u2)
    ax.set_title("Cercul corelatiilor pentru radacinile " + z1+ "/" + u1 + " si " + z2 + "/" + u2)
    ax.axhline(0)
    ax.axvline(0)
    x = np.arange(0, np.pi*2, 0.01)
    ax.plot(np.cos(x), np.sin(x))

    ax.scatter(tabel_r_xz[z1], tabel_r_xz[z2], label="X")
    ax.scatter(tabel_r_yu[u1], tabel_r_yu[u2], label="Y")

    for i in range(len(tabel_r_xz)):
        ax.text(tabel_r_xz[z1].iloc[i], tabel_r_xz[z2].iloc[i], tabel_r_xz.index[i])

    for i in range(len(tabel_r_yu)):
        ax.text(tabel_r_yu[u1].iloc[i], tabel_r_yu[u2].iloc[i], tabel_r_yu.index[i])

def corelograma(tabel_r, vmin=-1, titlu="Corelograma variabile observate - variabile canonice"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    heatmap(tabel_r, vmin=vmin, vmax=1, annot=True, ax=ax)

def biplot(tabel_z, z1, z2,tabel_u, u1, u2):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Biplot intre axele" + z1 + "/" + u1 +" si " + z2 + "/" + u2)
    ax.set_xlabel(z1 + "/" + u1)
    ax.set_ylabel(z2 + '/' + u2)
    ax.axhline(0)
    ax.axvline(0)

    ax.scatter(tabel_z[z1], tabel_z[z2], label="X")
    ax.scatter(tabel_u[u1], tabel_u[u2], label="Y")

    for i in range(len(tabel_z)):
        ax.text(tabel_z[z1].iloc[i], tabel_z[z2].iloc[i], tabel_u.index[i])
        ax.text(tabel_u[u1].iloc[i], tabel_u[u2].iloc[i], tabel_u.index[i])


def show():
    plt.show()