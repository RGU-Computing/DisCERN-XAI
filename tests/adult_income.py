from discern.discern_tabular import DisCERNTabular
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

def sklearn_test(attrib):
    train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'adult.data.csv'))
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'adult.test.csv'))

    data_df = train_df.append(test_df, ignore_index=True)
    data_df = data_df.replace({'salary': {' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1}})
    data_df = data_df[data_df['work-class'] != ' ?']
    data_df = data_df[data_df['occupation'] != ' ?']
    data_df = data_df[data_df['native-country'] != ' ?']

    d = dict(zip(data_df['work-class'].unique(), list(range(len(data_df['work-class'].unique())))))
    data_df['work-class'] = data_df['work-class'].map(d)
    d = dict(zip(data_df['occupation'].unique(), list(range(len(data_df['occupation'].unique())))))
    data_df['occupation'] = data_df['occupation'].map(d)
    d = dict(zip(data_df['education'].unique(), list(range(len(data_df['education'].unique())))))
    data_df['education'] = data_df['education'].map(d)
    d = dict(zip(data_df['marital-status'].unique(), list(range(len(data_df['marital-status'].unique())))))
    data_df['marital-status'] = data_df['marital-status'].map(d)
    d = dict(zip(data_df['native-country'].unique(), list(range(len(data_df['native-country'].unique())))))
    data_df['native-country'] = data_df['native-country'].map(d)
    d = dict(zip(data_df['race'].unique(), list(range(len(data_df['race'].unique())))))
    data_df['race'] = data_df['race'].map(d)
    d = dict(zip(data_df['relationship'].unique(), list(range(len(data_df['relationship'].unique())))))
    data_df['relationship'] = data_df['relationship'].map(d)
    d = dict(zip(data_df['sex'].unique(), list(range(len(data_df['sex'].unique())))))
    data_df['sex'] = data_df['sex'].map(d)

    data_df['age'] = (data_df['age'] - min(data_df['age'])) / (max(data_df['age']) - min(data_df['age']))
    data_df['fnlwgt'] = (data_df['fnlwgt'] - min(data_df['fnlwgt'])) / (max(data_df['fnlwgt']) - min(data_df['fnlwgt']))
    data_df['education-num'] = (data_df['education-num'] - min(data_df['education-num'])) / (max(data_df['education-num']) - min(data_df['education-num']))
    data_df['capital-gain'] = (data_df['capital-gain'] - min(data_df['capital-gain'])) / (max(data_df['capital-gain']) - min(data_df['capital-gain']))
    data_df['capital-loss'] = (data_df['capital-loss'] - min(data_df['capital-loss'])) / (max(data_df['capital-loss']) - min(data_df['capital-loss']))
    data_df['salary'] = (data_df['salary'] - min(data_df['salary'])) / (max(data_df['salary']) - min(data_df['salary']))
    data_df['hours-per-week'] = (data_df['hours-per-week'] - min(data_df['hours-per-week'])) / (max(data_df['hours-per-week']) - min(data_df['hours-per-week']))
    print("Reading data complete!")
    print("Columns: ", data_df.columns)

    df = data_df.copy()
    x = df.loc[:, df.columns != 'salary'].values
    y = df['salary'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    print("Train test split complete!")

    scalar = MinMaxScaler()
    x_train_norm = scalar.fit_transform(x_train)
    x_test_norm = scalar.transform(x_test)
    print("Data transform complete!")

    rfx = RandomForestClassifier(n_estimators=100)
    rfx.fit(x_train_norm, y_train)
    print("Training classifier complete: ", accuracy_score(y_test, rfx.predict(x_test_norm)))

    test_instance = x_test_norm[20]
    test_label = rfx.predict([x_test_norm[20]])[0]

    cat_indices = [1,3,5,6,7,8,9,13]
    imm_indices = [0,5,8,9,13]

    discern = DisCERNTabular(rfx, attrib)
    discern.init_data(x_train_norm, rfx.predict(x_train_norm), [c for c in df.columns if c!='salary'], ['<=50K', '>50K'], cat_feature_indices=cat_indices, immutable_feature_indices=imm_indices)
    cf, cf_label, s, p = discern.find_cf(test_instance, test_label)
    print('---------------------sklearn-'+attrib+'---------------------')
    print(cf, cf_label)
    print(test_instance, test_label)
    print("Sparsity: ",s, "Proximity: ", p)


def keras_test(attrib):
    train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'adult.data.csv'))
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'adult.test.csv'))

    data_df = train_df.append(test_df, ignore_index=True)
    data_df = data_df.replace({'salary': {' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1}})
    data_df = data_df[data_df['work-class'] != ' ?']
    data_df = data_df[data_df['occupation'] != ' ?']
    data_df = data_df[data_df['native-country'] != ' ?']

    d = dict(zip(data_df['work-class'].unique(), list(range(len(data_df['work-class'].unique())))))
    data_df['work-class'] = data_df['work-class'].map(d)
    d = dict(zip(data_df['occupation'].unique(), list(range(len(data_df['occupation'].unique())))))
    data_df['occupation'] = data_df['occupation'].map(d)
    d = dict(zip(data_df['education'].unique(), list(range(len(data_df['education'].unique())))))
    data_df['education'] = data_df['education'].map(d)
    d = dict(zip(data_df['marital-status'].unique(), list(range(len(data_df['marital-status'].unique())))))
    data_df['marital-status'] = data_df['marital-status'].map(d)
    d = dict(zip(data_df['native-country'].unique(), list(range(len(data_df['native-country'].unique())))))
    data_df['native-country'] = data_df['native-country'].map(d)
    d = dict(zip(data_df['race'].unique(), list(range(len(data_df['race'].unique())))))
    data_df['race'] = data_df['race'].map(d)
    d = dict(zip(data_df['relationship'].unique(), list(range(len(data_df['relationship'].unique())))))
    data_df['relationship'] = data_df['relationship'].map(d)
    d = dict(zip(data_df['sex'].unique(), list(range(len(data_df['sex'].unique())))))
    data_df['sex'] = data_df['sex'].map(d)

    data_df['age'] = (data_df['age'] - min(data_df['age'])) / (max(data_df['age']) - min(data_df['age']))
    data_df['fnlwgt'] = (data_df['fnlwgt'] - min(data_df['fnlwgt'])) / (max(data_df['fnlwgt']) - min(data_df['fnlwgt']))
    data_df['education-num'] = (data_df['education-num'] - min(data_df['education-num'])) / (max(data_df['education-num']) - min(data_df['education-num']))
    data_df['capital-gain'] = (data_df['capital-gain'] - min(data_df['capital-gain'])) / (max(data_df['capital-gain']) - min(data_df['capital-gain']))
    data_df['capital-loss'] = (data_df['capital-loss'] - min(data_df['capital-loss'])) / (max(data_df['capital-loss']) - min(data_df['capital-loss']))
    data_df['salary'] = (data_df['salary'] - min(data_df['salary'])) / (max(data_df['salary']) - min(data_df['salary']))
    data_df['hours-per-week'] = (data_df['hours-per-week'] - min(data_df['hours-per-week'])) / (max(data_df['hours-per-week']) - min(data_df['hours-per-week']))
    print("Reading data complete!")
    print("Columns: ", data_df.columns)

    df = data_df.copy()
    x = df.loc[:, df.columns != 'salary'].values
    y = df['salary'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    print("Train test split complete!")

    scalar = MinMaxScaler()
    x_train_norm = scalar.fit_transform(x_train)
    x_test_norm = scalar.transform(x_test)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(df['salary'].unique()), dtype='float32')
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(df['salary'].unique()), dtype='float32')      
    print("Data transform complete!")

    inputs = tf.keras.Input(shape=(x_train_norm.shape[-1],))
    hidden1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    outputs = tf.keras.layers.Dense(len(df['salary'].unique()), activation='softmax')(hidden2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model")

    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy'])

    model.fit(x_train_norm, y_train_cat, validation_data=(x_test, y_test_cat), batch_size=32, epochs=5, verbose=0)
    print("Training classifier complete: ", accuracy_score(y_test, model.predict(x_test_norm).argmax(axis=-1)))

    test_instance = x_test_norm[20]
    test_label = model.predict(np.array([x_test_norm[20]])).argmax(axis=-1)[0]

    cat_indices = [1,3,5,6,7,8,9,13]
    imm_indices = [0,5,8,9,13]

    discern = DisCERNTabular(model, attrib)
    discern.init_data(x_train_norm, model.predict(x_train_norm).argmax(axis=-1), [c for c in df.columns if c!='salary'], ['<=50K', '>50K'], cat_feature_indices=cat_indices, immutable_feature_indices=imm_indices)
    cf, cf_label, s, p = discern.find_cf(test_instance, test_label)
    print('---------------------keras-'+attrib+'---------------------')
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