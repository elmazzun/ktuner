#!/usr/bin/env python

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from sklearn.metrics import accuracy_score, classification_report
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

###############################################################################
# 1. Definisco modello tramite funzione build_model()
# 2. Creo il tuner RandomSearch()
# 3. Eseguo ipertuning tramite tuner.search()
# 4. Addestro il modello
###############################################################################

# Per ogni esperimento, concedo training al modello per max EPOCHS epoche
EPOCHS = 5
BATCH_SIZE = 32
# Se l'accuracy non migliora dopo 5 epochs, arresta il training del modello
# con quegli iperparametri e procedi al training con il prossimo set di
# iperparametri 
EARLY_STOPPING_PATIENCE = 3
MAX_TRIALS = 10

data = pd.read_csv(r"/home/amazzocchi/Downloads/train.csv") # Sistema path
for i in range(len(data["Sex"])):
    if data["Sex"][i] == "male":
        data["Sex"][i] = 0 # WARNING: A value is trying to be set on a copy of a slice from a DataFrame (??? cerca su google che vuol dire)
    else:
        data["Sex"][i] = 1

age_median = np.nanmedian(data["Age"])
data["Age"].fillna(age_median, inplace=True)

x = data[["Pclass","Sex","Age","Parch"]]
y = data["Survived"]
print(x.ndim)
x_train = x[:713]
x_test = x[713:]
y_train = y[:713]
y_test = y[713:]

x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')

# Crea un modello per l'ipertuning, definendo lo spazio di ricerca degli
# iperparametri oltre all'architettura del modello stesso
def build_model(hp):
    # Definisci lo spazio dei valori su cui testare questo modello
    ACTIVATIONS = ["relu", "tanh"]
    LEARNING_RATES = [1e-2, 1e-3, 1e-4] # 0.01, 0.001 oppure 0.0001
    TEST_WITH_HIDDEN_LAYERS = True
    # Aggiungo un numero di layer intermedi compreso
    # tra MIN_HIDDEN_LAYERS_NUM e MAX_HIDDEN_LAYERS_NUM
    MIN_HIDDEN_LAYERS_NUM = 1
    MAX_HIDDEN_LAYERS_NUM = 3
    # Per ogni layer intermedio, testo tale layer con un numero di nodi
    # compreso tra HIDDEN_LAYER_MIN_UNITS e HIDDEN_LAYER_MAX_UNITS
    HIDDEN_LAYER_MIN_UNITS = 32
    HIDDEN_LAYER_MAX_UNITS = 512
    # Testo con queste funzioni di attivazione
    active_func = hp.Choice("activation", ACTIVATIONS)
    learning_rate_func = hp.Choice("learning_rate", values=LEARNING_RATES)

    model = keras.Sequential()
    # Metto sempre un layer di input
    model.add(layers.Input(shape=(4,)))

    if TEST_WITH_HIDDEN_LAYERS:    
        for layer_i in range(hp.Int('num_layers', MIN_HIDDEN_LAYERS_NUM, MAX_HIDDEN_LAYERS_NUM)):
            # "units_X" è il numero di nodi del layer intermedio numero X
            # Es: "units_0" è il numero di nodi nel primo layer intermedio
            units_func = hp.Int("units_" + str(layer_i),
                                min_value=HIDDEN_LAYER_MIN_UNITS,
                                max_value=HIDDEN_LAYER_MAX_UNITS, step=32)
            # Aggiungo layer intermedio con numero di nodi e funzione di
            # attivazione variabili: questo layer è Dense ma si possono
            # aggiungere layer di altra natura
            model.add(layers.Dense(units=units_func, activation=active_func))

    model.add(layers.Dense(1, activation='sigmoid'))

    # Per ora testo solo con ottimizzatore Adam ma
    # si possono testare multipli ottimizzatori
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_func)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

################### Inizio
# Si possono testare più tuners contemporaneamente, iterando sull'array
# "tuners": per ora testo solo con il tuner RandomSearch
# tuners = [
#     RandomSearch(build_model, objective='val_accuracy',
#                  max_trials=10, executions_per_trial=3),
#     Hyperband(build_model, objective="val_accuracy",
#               max_epochs=EPOCHS, factor=3, seed=42),
#     BayesianOptimization(build_model, objective="val_accuracy",
#                          max_trials=10, seed=42)
# ]

# Creo il tuner degli iperparametri 
tuner = RandomSearch(
    build_model, # chiamo funzione definita sopra
    objective='val_accuracy',
    max_trials=MAX_TRIALS,
    executions_per_trial=3)
    # directory='my_dir',
    # project_name='test')

# Definisco una funzione per impedire al modello di overfittare oppure
# di spendere troppo tempo a fare training con minimi guadagni: se dunque
# il training del modello ha scarse performance, lo arresto e proseguo
# il training con altri iperparametri
stop_early = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)

print(">>> Riassunto dello spazio degli iperparametri sui cui addestrare il modello")
tuner.search_space_summary()

print(">>> Effettuo ricerca sugli iperparametri...")
tuner.search(x=x_train, y=y_train,
             batch_size=BATCH_SIZE, epochs=EPOCHS,
             validation_data=(x_test, y_test),
             callbacks=[stop_early],
             verbose=0) # Se vuoi stampare output di tuner.search, metti verbose=1

print(">>> Stampo migliori iperparametri trovati finora")
tuner.results_summary(num_trials=1)

# Prendo i migliori iperparametri
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
>>> Ricerca degli iperparametri completa.
>>> Il learning rate ottimale per l'ottimizzatore è {best_hps.get('learning_rate')}.
>>> Il numero ottimale di layers intermedi è {best_hps.get('num_layers')}.""")

for layer in range(best_hps.get('num_layers')):
    print(f">>> Il numero ottimale di nodi nel layer {layer} è {best_hps.get('units_'+str(layer))}.")

print(">>> Trovo il numero ottimale di epochs con cui addestrare il modello usando"
"gli iperparametri trovati da search")
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=EPOCHS, verbose=1)

print(">>> Ricreo l'ipermodello e lo addestro con il numero ottimale di epoche"
"ottenuto sopra")
val_acc_per_epoch = history.history['accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

# Miglior numero di epochs
print('>>> Best epoch: %d' % (best_epoch,))

# Ricrea il modello con gli iperparametri trovati prima...
hypermodel = tuner.hypermodel.build(best_hps)
# ...ma il train stavolta dura "best_epoch"
history = hypermodel.fit(x_train, y_train, epochs=best_epoch)

# evaluate() fornisce loss e 
print(">>> Infine valuto l'ipermodello ottenuto sui dati di test")
eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result) 
