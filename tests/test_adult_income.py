from discern.discern_tabular import DisCERNTabular
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

def test_adult_income():
    data_df = pd.read_csv('adult_income.csv')
    df = data_df.copy()
    print("Reading data complete!")

    x = df.loc[:, df.columns != 'salary'].values
    y = df['salary'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    print("Train test split complete!")

    scalar = MinMaxScaler()
    x_train= scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)
    print("Data transform complete!")

    rfx = RandomForestClassifier(n_estimators=500)
    rfx.fit(x_train, y_train)
    print("Training classifier complete!")
    print(accuracy_score(y_test, rfx.predict(x_test)))

    # indices = np.random.choice(x_test.shape[0], 2, replace=False)
    x_test = x_test[:10]
    y_test = rfx.predict(x_test[:10])

    # print(x_test.shape)
    # print(y_test.shape)

    sparsity = []
    proximity = []
    discern = DisCERNTabular(rfx, 'LIME', 'Q')
    discern.init_data(x_train, y_train, [c for c in df.columns if c!='salary'], ['<=50K', '>50K'], cat_feature_indices=[])

    for idx in range(len(x_test)):
        cf, s, p = discern.find_cf(x_test[idx], y_test[idx])
        sparsity.append(s)
        proximity.append(p)

    _sparsity = sum(sparsity)/len(sparsity)
    _proximity = sum(proximity)/(len(proximity)*_sparsity)
    print(_sparsity)
    print(_proximity)


def test_adult_income_cat():
    train_df = pd.read_csv('adult.data.csv')
    test_df = pd.read_csv('adult.test.csv')

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
    
    df = data_df.copy()
    x = df.loc[:, df.columns != 'salary'].values
    y = df['salary'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    print("Train test split complete!")

    scalar = MinMaxScaler()
    x_train= scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)
    print("Data transform complete!")
    
    rfx = RandomForestClassifier(n_estimators=500)
    rfx.fit(x_train, y_train)
    print(accuracy_score(y_test, rfx.predict(x_test)))
    print("Training classifier complete!")

    # indices = np.random.choice(x_test.shape[0], 20, replace=False)
    x_test = x_test[:10]
    y_test = rfx.predict(x_test[:10])

    cat_indices = [1,3,5,6,7,8,9,13]

    sparsity = []
    proximity = []
    discern = DisCERNTabular(rfx, 'LIME', 'Q')
    discern.init_data(x_train, y_train, [c for c in df.columns if c!='salary'], ['<=50K', '>50K'], cat_feature_indices=cat_indices)

    for idx in range(len(x_test)):
        cf, s, p = discern.find_cf(x_test[idx], y_test[idx])
        print(s)
        print(p)
        sparsity.append(s)
        proximity.append(p)

    _sparsity = sum(sparsity)/len(sparsity)
    _proximity = sum(proximity)/(len(proximity)*_sparsity)
    print(_sparsity)
    print(_proximity)


def test_adult_income_svm():
    data_df = pd.read_csv('adult_income.csv')
    df = data_df.copy()
    print("Reading data complete!")

    x = df.loc[:, df.columns != 'salary'].values
    y = df['salary'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    print("Train test split complete!")

    scalar = MinMaxScaler()
    x_train= scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)
    print("Data transform complete!")

    svm = SVC(probability=True)
    svm.fit(x_train, y_train)
    print("Training classifier complete!")
    print(accuracy_score(y_test, svm.predict(x_test)))

    # indices = np.random.choice(x_test.shape[0], 2, replace=False)
    x_test = x_test[:10]
    y_test = svm.predict(x_test[:10])

    # print(x_test.shape)
    # print(y_test.shape)

    sparsity = []
    proximity = []
    discern = DisCERNTabular(svm, 'LIME', 'Q')
    discern.init_data(x_train, y_train, [c for c in df.columns if c!='salary'], ['<=50K', '>50K'], cat_feature_indices=[])

    for idx in range(len(x_test)):
        cf, s, p = discern.find_cf(x_test[idx], y_test[idx])
        sparsity.append(s)
        proximity.append(p)

    _sparsity = sum(sparsity)/len(sparsity)
    _proximity = sum(proximity)/(len(proximity)*_sparsity)
    print(_sparsity)
    print(_proximity)

