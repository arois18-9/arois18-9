# Métricas de desempeño

def model_metrics(model, X_test, y_test):
 
  y_test_pred = model.predict(X_test)
  print(f"Accuracy = {accuracy_score(y_test,y_test_pred)}")
  print(f"Recall = {recall_score(y_test,y_test_pred)}")

  #Las métricas F1, precision and recall requieren que se establezca la convención de cuál es la clase positiva (1)
  print(f"F1 score = {f1_score(y_test,y_test_pred)}")
  
  disp = plot_confusion_matrix(model, X_test, y_test, display_labels=model.classes_,
                              cmap=plt.cm.Blues, 
                              normalize='true')
  disp.ax_.set_title('Matriz de confusión')
  plt.show()

def confusion_matrix_labels(model, X_test, y_test):
    '''
    Esta función permite graficar la matriz de confusión con los labels en valor predicho correspondientes a: 
    - Verdareros positivos
    - Falsos positivos
    - Verdaderos negativos
    - Falsos negativos
    '''
    predicciones = model.predict(X_test)

    df_predicciones = pd.DataFrame({"True": y_test, "Pred": predicciones})
    df_predicciones.head()

    cf_matrix = confusion_matrix(y_test, predicciones)

    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

# Función para obtener la métrica de validación cruzada del modelo

def cross_val(model, X, y):
 
  kf = KFold(n_splits=10)
  scores = cross_val_score(model, X, y, cv=kf, scoring="f1")
  print(f"Metricas cross_validation \n{scores.round(2)}")
  print("Media de cross_validation", scores.mean().round(2))

# Función para obtener la métrica del AUC; área bajo la curva ROC

def roc_auc_metrics(model, name_model, X_test, y_test):

    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = model.predict_proba(X_test)[:, 1]
    # Calculamos el AUC
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # Imprimimos en pantalla
    print('Sin entrenar: ROC AUC=%.3f' % (ns_auc))
    print(name_model,'ROC AUC=%.3f' % (lr_auc))
    # Calculamos las curvas ROC
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # Pintamos las curvas ROC
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=name_model)
    # Etiquetas de los ejes
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.legend()
    plt.show()

# Dataset Entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X_res,y_res, test_size=0.3, random_state=12)
print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)

y_train.value_counts()
y_test.value_counts()

# Modelos
# Modelo Random Forest Classifier

model_randomforest = RandomForestClassifier(max_depth=2, random_state=0, class_weight = 'balanced')
model_randomforest.fit(X_train, y_train)

parameteres = {'n_estimators':[40,50,80], 'max_depth':[4,5,6], 'class_weight':['None','balanced']}

grid = GridSearchCV(model_randomforest, param_grid=parameteres, cv=10)
kf_hiper = KFold(n_splits=10)
model_randomforest = grid.fit(X_train, y_train).best_estimator_

model_randomforest

model_metrics(model=model_randomforest, X_test=X_test, y_test=y_test)

Accuracy = 0.8993516659606211
Recall = 0.8608809339667808
F1 score = 0.8953471670289473

confusion_matrix_labels(model_randomforest, X_test, y_test)

cross_val(model_randomforest, X_test, y_test)
Metricas cross_validation 
[0.88 0.88 0.88 0.88 0.88 0.88 0.89 0.9  0.89 0.88]
Media de cross_validation 0.88


roc_auc_metrics(model_randomforest,"Random Forest Classifier",X_test, y_test)
Sin entrenar: ROC AUC=0.500
Random Forest Classifier ROC AUC=0.966


importances = model_randomforest.feature_importances_
features = pd.Series(importances, index=X_train.columns)
plt.figure(figsize=(10, 25))
features.plot(kind="barh")
plt.show()


# Modelo de Red Neuronal con Autoencoder

# Para un mejor ajuste en este modelo, se escalan los datos
t = MinMaxScaler()
t.fit(X_train)
X_train_RN = t.transform(X_train)
X_test_RN = t.transform(X_test)

# Se define el encoder
n_inputs = X_train.shape[1]
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)

# Se define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# Capa de salida
output = Dense(n_inputs, activation='linear')(d)
# Se define el modelo del autoencoder
model_RN = Model(inputs=visible, outputs=output)

# Se realiza la compilacón del modelo de autoencoder
model_RN.compile(optimizer='adam', loss='mse')

model_RN.summary()
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 86)]              0         
_________________________________________________________________
dense (Dense)                (None, 172)               14964     
_________________________________________________________________
batch_normalization (BatchNo (None, 172)               688       
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 172)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 86)                14878     
_________________________________________________________________
batch_normalization_1 (Batch (None, 86)                344       
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 86)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 86)                7482      
_________________________________________________________________
dense_3 (Dense)              (None, 86)                7482      
_________________________________________________________________
batch_normalization_2 (Batch (None, 86)                344       
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 86)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 172)               14964     
_________________________________________________________________
batch_normalization_3 (Batch (None, 172)               688       
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 172)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 86)                14878     
=================================================================
Total params: 76,712
Trainable params: 75,680
Non-trainable params: 1,032
_________________________________________________________________


# Se entrena el modelo de autoencoder para reconstruir la entrada
history = model_RN.fit(X_train_RN, X_train_RN, epochs=10, batch_size=16, verbose=2, validation_data=(X_test_RN,X_test_RN))

Epoch 1/10
15836/15836 - 41s - loss: 0.0135 - val_loss: 0.0040
Epoch 2/10
15836/15836 - 39s - loss: 0.0070 - val_loss: 0.0023
Epoch 3/10
15836/15836 - 39s - loss: 0.0049 - val_loss: 0.0021
Epoch 4/10
15836/15836 - 40s - loss: 0.0034 - val_loss: 0.0015
Epoch 5/10
15836/15836 - 40s - loss: 0.0026 - val_loss: 0.0011
Epoch 6/10
15836/15836 - 39s - loss: 0.0022 - val_loss: 0.0011
Epoch 7/10
15836/15836 - 41s - loss: 0.0020 - val_loss: 0.0013
Epoch 8/10
15836/15836 - 39s - loss: 0.0018 - val_loss: 0.0011
Epoch 9/10
15836/15836 - 40s - loss: 0.0016 - val_loss: 7.6036e-04
Epoch 10/10
15836/15836 - 44s - loss: 0.0015 - val_loss: 7.6939e-04


# Se grafica la función de pérdida
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()




