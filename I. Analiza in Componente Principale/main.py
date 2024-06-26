import pandas as pd
import numpy as np
from functii import *
from sklearn.decomposition import PCA

set_date = pd.read_csv("mortalitate_ro.csv", index_col=0)
nan_replace(set_date)
variabile = list(set_date.columns)[1:]
x = set_date[variabile]
n,m = np.shape(x)
etichete = ["C" + str(i+1) for  i in range(m)]

#1. Standardizare date
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

#2. Antrenare model
model_pca = PCA()
model_pca.fit(x)

#3. Varianta componente
alpha = model_pca.explained_variance_ * (n - 1)/n
tabel_alpha = tabelare_varianta(alpha, etichete)
tabel_alpha.to_csv("out/Varianta.csv")

#4. Plot varianta cu evidentierea criteriilor
criterii = calcul_criterii(alpha)
plot_varianta(alpha, criterii)

#5.Corelatii factoriale (corelatii variabile - componente principale)

#1. Calcul componente si scoruri
c = model_pca.transform(x) #componente
s = c / np.sqrt(alpha) #scoruri

tabel_c = pd.DataFrame(c, set_date.index, etichete)
tabel_s = pd.DataFrame(s, set_date.index, etichete)
tabel_c.to_csv("out/Componente.csv")
tabel_s.to_csv("out/Scoruri.csv")

#2. Corelatii factoriale
r = np.corrcoef(x, c, rowvar=False)[:m, m:]
tabel_r = pd.DataFrame(r, variabile, etichete)
tabel_r.to_csv("out/Corelatii_Factoriale.csv")

#3. Corelograma corelatii
corelograma(tabel_r)

#4. Cercul corelatiilor factoriale
cercul_corelatiilor(tabel_r, "C1", "C2")

#5. Trasare plot componente / scoruri
plot_scoruri(tabel_s, "C1", "C2")

#6. Calcul cosinusuri -> calitatea reprezentarii fiecarei instante pe axa
c2 = c*c
cosin = (c2.T / np.sum(c2, axis=1)).T
tabel_cosin = pd.DataFrame(cosin, set_date.index, etichete)
tabel_cosin.to_csv("out/Cosinusuri.csv")

#7. Calcul contributii -> contributia instantelor la varianta axelor
contrib = c2 / np.sum(c2, axis=0)
tabel_contrib = pd.DataFrame(contrib, set_date.index, etichete)
tabel_contrib.to_csv("out/Contributii.csv")

#8. Comunalitati -> varianta comuna explicata de un numar de componente principale
r2 = r*r
comm = np.cumsum(r2, axis=1)
tabel_comm = pd.DataFrame(comm, variabile, etichete)
tabel_comm.to_csv("out/Comunalitati.csv")

corelograma(tabel_comm, vmin=0, titlu="Corelograma comunalitati")

show()
