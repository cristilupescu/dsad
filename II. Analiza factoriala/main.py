import pandas as pd
import numpy as np
from functii import *
import factor_analyzer as fact

set_date = pd.read_csv("Teritorial_2022.csv", index_col=0)
nan_replace(set_date)
variabile = list(set_date.columns)[3:]
x = set_date[variabile]
m = len(variabile)
etichete = ["F" + str(i+1) for i in range(m)]

#1. Testul Bartlett -> compara matricea de corelatie a datelor cu matricea identitate
#H0 -> nu exista factori
#H1 -> avem cel putin un factor
test_bartlett = fact.calculate_bartlett_sphericity(x)
if test_bartlett[1] > 0.01: #daca e mai mic respingem H0 acceptam H1
    print("Nu exista factori comuni")
    exit(0)

#2. Indecsi KMO -> ne arata atat pentru fiecare variabila, cat si pentru intregul set daca sunt potrivite pentru analiza factoriala
indecsi_kmo = fact.calculate_kmo(x)

if all(indecsi_kmo[0] < 0.6):
    print("Nu exista factori comuni")
    exit(0)

tabel_kmo = pd.DataFrame(data={
    "Indecsi KMO": np.append(indecsi_kmo[0], indecsi_kmo[1])
}, index=[variabile + ["Index Global"]])

tabel_kmo.to_csv("out/Indecsi_KMO.csv")

#3. Antrenare model -> Putem fara rotatie sau cu rotatie
# rotatie (varimax) -> sistemul de axe e rotit astfel incat fiecare variabila sa fie asociata cu un singur factor
model_fact = fact.FactorAnalyzer(n_factors=m, rotation=None)
# model_fact = fact.FactorAnalyzer(n_factors=m, rotation="varimax")
model_fact.fit(x)

#4. Calcul varianta factori
alpha = model_fact.get_factor_variance()
tabel_alpha = tabelare_varianta(alpha)
tabel_alpha.to_csv("out/Varianta.csv")

#5.Calcul corelatii factoriale -> corelatii intre variabile si factori
r = model_fact.loadings_
tabel_r = pd.DataFrame(r, variabile, etichete)
tabel_r.to_csv("out/Corelatii_Factoriale.csv")

#6 Corelograma corelatii factoriale
corelograma(tabel_r)

#7. Cercul corelatiilor
plot_corelatii(tabel_r, "F1", "F2")

#8. Calcul scoruri
s = model_fact.transform(x)
tabel_s = pd.DataFrame(s, set_date.index, etichete)
tabel_s.to_csv("out/Scoruri.csv")

#9.Trasare plot scoruri
plot_scoruri(tabel_s, "F1", "F2")


#10. Calcul comunalitati -> comunalitatile reprezinta varianta comuna explicata de un numar de factori
comm = model_fact.get_communalities()
tabel_comm = pd.DataFrame(
    data={
        "Comunalitati": np.round(comm, 3)
    }, index=variabile

)

tabel_comm.to_csv("out/Comunalitati.csv")

#11.Varianta factori specifici -> sunt factorii care sunt specifici fiecarei variabile
psi = model_fact.get_uniquenesses()
tabel_psi = pd.DataFrame(data={
    "Varianta specifica": psi
}, index=variabile)

tabel_psi.to_csv("out/Varianta_Specifica.csv")



#12. Corelograma comunalitati si varianta specifica si indecsi kmo
corelograma(tabel_comm, vmin=0,titlu="Corelograma comunalitati")
corelograma(tabel_psi, vmin=0, titlu="Varianta specifica")
corelograma(tabel_kmo, vmin=0, titlu="Corelograma kmo")



show()
