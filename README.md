## covid death prediction using ML
running main.py will train a model to predict whether a covid patient will die with an accuracy of 91% (as shown in the classification report).
#### how it works
the data will be pre processed and then will be fed to an MLPClassifier which has been fine tuned using gridsearch CV
### data
i got the covid statistics from the mexican government: https://datos.gob.mx/busca/dataset/informacion-referente-a-casos-covid-19-en-mexico
