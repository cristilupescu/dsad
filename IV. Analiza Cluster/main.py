import pandas as pd
import numpy as np
from functii import *
import scipy.cluster.hierarchy as hclust
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

set_date = pd.read_csv("ADN_Tari.csv", index_col=0)
variabile = list(set_date.columns)[1:]
x = set_date[variabile]

#1.Matrice de ierarhie
h = hclust.linkage(x, method="ward")


#2. Calcul partitie optimala
k_opt, threshold_opt, partitie_opt = calcul_partitie(h)

tabel_partitie_optimala = pd.DataFrame(data={
        "Partitie": partitie_opt
    }, index=set_date.index
)

tabel_partitie_optimala.to_csv("out/Partitie_Optimala.csv")

#3. Partitie oarecare
k, threshold_k, partitie_k = calcul_partitie(h, 7)


#4. Calcul indecsi shillouette

#la nivel de partitie -> media indecsilor shilouette la nivel de instanta
indecsi_sihlouette_partitie = silhouette_score(x, partitie_opt)

#la nivel de instante
indecsi_silhouette_instante = silhouette_samples(x, partitie_opt)


tabel_partitie_optimala = pd.DataFrame(data={
    "Partitie": partitie_opt,
    "Indecsi Sihlouette": indecsi_silhouette_instante
}, index=set_date.index)

tabel_partitie_optimala.to_csv("out/Partitie_Optimala.csv")

#5. Plot dendograma
dendograma(h, threshold_opt, set_date.index)
dendograma(h, threshold_k, set_date.index)

#6. Trasare plot Shilouette
plot_silh(x, partitie_opt)
plot_silh(x, partitie_k)

#8. Histograme clusteri pentru fiecare variabila onservata
histograma(set_date, "I1", partitie_opt)

#9. Plot partitie in axele principale
model_pca = PCA(2)
model_pca.fit(x)
z = model_pca.transform(x)
etichete = ["Z1","Z2"]
tabel_z = pd.DataFrame(z, set_date.index, etichete)

plot_axe_principale(tabel_z, "Z1", "Z2", partitie_opt)


show()
