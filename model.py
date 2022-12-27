from sklearn.neural_network import MLPClassifier

# after fine tuning the model with gridsearch cv these were the best performing parameters
model = MLPClassifier(alpha=0.0000001,
                      hidden_layer_sizes=14,
                      solver='lbfgs')

