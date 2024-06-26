import pandas as pd
import numpy as np
from functii import *
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import normalize

tabel1 = pd.read_csv("Prezenta_vot_judete_p.csv", index_col=0)
tabel2 = pd.read_csv("Vot_Senat_Jud_Proc.csv", index_col=0)

nan_replace_t(tabel1)
nan_replace_t(tabel2)

variabile1 = list(tabel1.columns)
variabile2 = list(tabel2.columns)

x = tabel1[variabile1].values
y = tabel2[variabile2].values

tabel = tabel1.merge(right=tabel2, left_index=True, right_index=True)

n = len(tabel) #numarul de linii din tabelul de jonctiune
p = len(variabile1) #numarul de variabile din primul tabel
q = len(variabile2) #numarul de variabile din al doilea tabel
m = min(p, q) #numarul minim de variabile

#1. Construire model
model_cca = CCA(n_components=m)
model_cca.fit(x, y)

#2. Scoruri canonice
z, u = model_cca.transform(x,y)
z = normalize(z, axis=0)
u = normalize(u, axis=0)

etichete_z = ["Z" + str(i+1) for i in range(m)]
etichete_u = ["U" + str(i+1) for i in range(m)]
etichete_rad = ["Rad" + str(i+1) for i in range(m)]

tabel_z = pd.DataFrame(z, tabel1.index, etichete_z)
tabel_u = pd.DataFrame(u, tabel2.index, etichete_u)

tabel_z.to_csv("out/Scoruri_Z.csv")
tabel_u.to_csv("out/Scoruri_U.csv")

#2. Calculul corelatiilor canonice
r = np.diag(np.corrcoef(z, u, rowvar=False)[:m, m:])
tabel_r = pd.DataFrame(data={
    "Corelatii": r
}, index=etichete_rad)

tabel_r.to_csv("out/Corelatii_Canonice.csv")

#3. Determinarea relevantei radacinilor cu testul bartlett
r2 = r*r
p_values = test_bartlett(r2, n, p, q, m)

tabel_semnificatie = pd.DataFrame(data={
    "R": np.round(r, 5),
    "R2": np.round(r2, 5),
    "P_Values": np.round(p_values, 5)
}, index=etichete_rad)

tabel_semnificatie.to_csv("out/Tabel_Semnificatie.csv")

#5. Numarul de radacini semnificative
nr_radacini_semnificative = np.where(p_values > 0.01)[0][0]
print(nr_radacini_semnificative)

#6. Corelatii variabile observate - variabile canonice
r_xz = np.corrcoef(x, z, rowvar=False)[:p,p:]
r_yu = np.corrcoef(y, u, rowvar=False)[:q,q:]

tabel_r_xz = pd.DataFrame(r_xz, variabile1, etichete_z)
tabel_r_yu = pd.DataFrame(r_yu, variabile2, etichete_u)


tabel_r_xz.to_csv("out/Corelatii_X_Z.csv")
tabel_r_yu.to_csv("out/Coerlatii_Y_U.csv")

#7. Calcul varianta explicita si redundanta informationala
vx = np.sum(r_xz[:,:m] * r_xz[:, :m], axis=0)
vy = np.sum(r_yu[:, :m] * r_yu[:, :m], axis=0)

rx = vx*r2[:m]
ry = vy*r2[:m]


tabel_varianta_redundanta = pd.DataFrame(data={
    "vx": vx,
    "vy": vy,
    "vx(%)": vx*100/p,
    "vy(%)": vy*100/q,
    "rx:": rx,
    "ry": ry,
    "rx(%)": rx*100/p,
    "ry(%)": ry*100/q
}, index=etichete_rad)

tabel_varianta_redundanta.to_csv("out/Varianta_Redundanta.csv")


#1. Plot corelatii variabile observate variabile canonice (cercul corelatilor)
cercul_corelatiilor(tabel_r_xz, "Z1", "Z2", tabel_r_yu, "U1", "U2")

#2. Corelograma corelatii variabile oservate  - variabile canonice
corelograma(tabel_r_xz, vmin=-1, titlu="Corelograma X - Z")
corelograma(tabel_r_yu, vmin=-1, titlu="Corelograma Y - U")

#3.Plot instante in spatiul celor 2 variabile (Biplot)
biplot(tabel_z, "Z1", "Z2", tabel_u, "U1", "U2")



show()
