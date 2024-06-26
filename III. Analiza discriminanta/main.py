import pandas as pd
import numpy as np
from functii import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

set_date = pd.read_csv("hernia.csv", index_col=0)
nan_replace(set_date)

variabile = list(set_date.columns)
predictori = variabile[:-1]
tinta = variabile[-1]

#1. Impartire model
x_train, x_test, y_train, y_test = train_test_split(set_date[predictori],
                                                    set_date[tinta],
                                                    test_size=0.4)


#A. Model liniar
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(x_train, y_train)

#1. Scoruri discriminante pe modelul liniar -> reprezinta axele discriminante
# Axele discriminante sunt noi variabile predictor, asemanatoare ca la analiza in componente principale
# Aceste axe sunt combinatii liniare de variabile observate si necorelate 2 cate 2
clase = model_lda.classes_
q = len(clase)
m = len(predictori)
nr_discriminatori = min(q-1, m)
etichete_z = ["Z" + str(i+1) for i in range(nr_discriminatori)]
z = model_lda.transform(x_test)
tabel_z = pd.DataFrame(z, x_test.index, etichete_z)
tabel_z.to_csv("out/Scoruri_Discriminante.csv")

#centrii discriminare
tabel_g = tabel_z.groupby(by=y_test.values).mean()
print(tabel_g)

#2. Plot-uri ( Plot instante si plot distributii)

#Plot instante
plot_instante(tabel_z, "Z1", "Z2", y_test)

#Plot distributii
plot_distributii(tabel_z, "Z1", y_test)
plot_distributie(z, y_test)

#3. Predictie pe setul de testare model liniar
predictie_lda_test = model_lda.predict(x_test)

#4. Evaluare model liniar
tabel_confuzie_lda, tabel_indicatori_lda = evaluare_model(y_test, predictie_lda_test, clase)
tabel_confuzie_lda.to_csv("out/Matrice_Confuzie_Lda.csv")
tabel_indicatori_lda.to_csv("out/Indicatori_Lda.csv")

tabel_erori = pd.DataFrame(data={
    "Tinta" : y_test,
    "Predictie": predictie_lda_test
}, index=x_test.index)

tabel_erori.to_csv("out/Tabel_Erori.csv")

#5. Predictia pe setul de aplicare
x_apply = pd.read_csv("hernia_apply.csv", index_col=0)
predictie_lda_apply = model_lda.predict(x_apply[predictori])

x_apply["Predictie_Model_Liniar"] = predictie_lda_apply

x_apply.to_csv("out/Predictie_Model_Liniar.csv")


#B. Model Bayesian
model_bayes = GaussianNB()
model_bayes.fit(x_train, y_train)

#1. Predictie model bayesian setul de test
predictie_bayes_test = model_bayes.predict(x_test)

tabel_predictie_test_bayes = pd.DataFrame(data={
    "Predictie": predictie_bayes_test
}, index=x_test.index)

tabel_predictie_test_bayes.to_csv("out/Predictie_Test_Bayes.csv")

#2. Evaluare model bayesian
tabel_confuzie_bayes, tabel_indicatori_bayes = evaluare_model(y_test, predictie_bayes_test, clase)
tabel_confuzie_bayes.to_csv("out/Matrice_Confuzie_Bayes.csv")
tabel_indicatori_bayes.to_csv("out/Indicatori_Bayes.csv")

#3. Predictie pe setul de aplicare model bayesian
predictie_bayes_apply = model_bayes.predict(x_apply[predictori])

x_apply["Predictie_Model_Bayes"] = predictie_bayes_apply

x_apply.to_csv("out/Predictie_Final.csv")




show()


