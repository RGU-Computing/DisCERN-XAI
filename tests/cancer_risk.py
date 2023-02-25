from discern.discern_tabular import DisCERNTabular
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import numpy as np
import tensorflow as tf

def sklearn_test(attrib):
    data_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'lung_cancer.csv'))
    data_df = data_df.replace({'Level': {'Low': 0, 'Medium': 1, 'High': 2}})
    data_df = data_df.replace({'Gender': {2: 0}})
    data_df = data_df.replace({'Alcohol use': {2: 0}})
    data_df = data_df.replace({'Dust Allergy': {2: 0}})
    data_df = data_df.replace({'Smoking': {2: 0}})
    data_df = data_df.replace({'Chest Pain': {2: 0}})
    data_df = data_df.replace({'Fatigue': {2: 0}})
    data_df = data_df.replace({'Shortness of Breath': {2: 0}})
    data_df = data_df.replace({'Wheezing': {2: 0}})
    data_df = data_df.replace({'Swallowing Difficulty': {2: 0}})
    data_df = data_df.replace({'Cough': {2: 0}})
    data_df = data_df.replace({'chronic Lung Disease': {2: 0}})
    print("Reading data complete!")

    df = data_df.copy()
    x = df.loc[:, df.columns != 'Level'].values
    y = df['Level'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
    print("Train test split complete!")
    
    scaler = MinMaxScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)
    print("Data transform complete!")

    rfx = RandomForestClassifier(n_estimators=100)
    rfx.fit(x_train, y_train)
    print(accuracy_score(y_test, rfx.predict(x_test)))
    print("Training classifier complete!")

    test_instance = x_test_norm[10]
    test_label = rfx.predict([x_test_norm[10]])[0]
    cat_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    imm_indices = [0, 1, 2]
    discern = DisCERNTabular(rfx, attrib)
    discern.init_data(x_train_norm, y_train, [c for c in df.columns if c!='Level'], ['Low', 'Medium', 'High'], cat_feature_indices=cat_indices, immutable_feature_indices=imm_indices)

    cf, cf_label, s, p = discern.find_cf(test_instance, test_label, cf_label=0)
    print('---------------------sklearn-'+attrib+'---------------------')
    print(cf, cf_label)
    print(test_instance, test_label)
    print("Sparsity: ",s, "Proximity: ", p)


def keras_test(attrib):
    data_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'lung_cancer.csv'))
    data_df = data_df.replace({'Level': {'Low': 0, 'Medium': 1, 'High': 2}})
    data_df = data_df.replace({'Gender': {2: 0}})
    data_df = data_df.replace({'Alcohol use': {2: 0}})
    data_df = data_df.replace({'Dust Allergy': {2: 0}})
    data_df = data_df.replace({'Smoking': {2: 0}})
    data_df = data_df.replace({'Chest Pain': {2: 0}})
    data_df = data_df.replace({'Fatigue': {2: 0}})
    data_df = data_df.replace({'Shortness of Breath': {2: 0}})
    data_df = data_df.replace({'Wheezing': {2: 0}})
    data_df = data_df.replace({'Swallowing Difficulty': {2: 0}})
    data_df = data_df.replace({'Cough': {2: 0}})
    data_df = data_df.replace({'chronic Lung Disease': {2: 0}})
    print("Reading data complete!")

    df = data_df.copy()
    x = df.loc[:, df.columns != 'Level'].values
    y = df['Level'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
    print("Train test split complete!")
    
    scaler = MinMaxScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(df['Level'].unique()), dtype='float32')
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(df['Level'].unique()), dtype='float32')      
    print("Data transform complete!")

    inputs = tf.keras.Input(shape=(x_train_norm.shape[-1],))
    hidden1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    outputs = tf.keras.layers.Dense(len(df['Level'].unique()), activation='softmax')(hidden2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model")

    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy'])

    model.fit(x_train_norm, y_train_cat, validation_data=(x_test, y_test_cat), batch_size=32, epochs=5, verbose=0)
    print("Training classifier complete: ", accuracy_score(y_test, model.predict(x_test_norm).argmax(axis=-1)))

    test_instance = x_test_norm[12]
    test_label = model.predict(np.array([x_test_norm[12]])).argmax(axis=-1)[0]

    cat_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    imm_indices = [0, 1, 2]
    discern = DisCERNTabular(model, attrib)
    print('labels', set( model.predict(x_train_norm).argmax(axis=-1)))
    discern.init_data(x_train_norm, model.predict(x_train_norm).argmax(axis=-1), [c for c in df.columns if c!='Level'], ['Low', 'Medium', 'High'], cat_feature_indices=cat_indices, immutable_feature_indices=imm_indices)

    cf, cf_label, s, p = discern.find_cf(test_instance, test_label, cf_label=0)
    print('---------------------sklearn-'+attrib+'---------------------')
    print(cf, cf_label)
    print(test_instance, test_label)
    print("Sparsity: ",s, "Proximity: ", p)

try:
    sklearn_test('LIME')
except:
    None
try:
    sklearn_test('SHAP')
except:
    None
try:
    keras_test('LIME')
except:
    None
try:
    keras_test('SHAP')
except:
    None
try:
    keras_test('IntG')
except:
    None