from discern import discern_tabular
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def test_adult_income_data():
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

    indices = np.random.choice(x_test.shape[0], 100, replace=False)
    x_test = x_test[indices]
    y_test = y_test[indices]

    print(x_test.shape)
    print(y_test.shape)

    sparsity = []
    proximity = []
    discern = discern_tabular.DisCERN(rfx, 'LIME', 'Q')
    discern.init_data(x_train, y_train, [c for c in df.columns if c!='salary'], ['<=50K', '>50K'])

    for idx in range(100):
        cf, s, p = discern.find_cf(x_test[idx], y_test[idx])
        sparsity.append(s)
        proximity.append(p)

    _sparsity = sum(sparsity)/len(sparsity)
    _proximity = sum(proximity)/(len(proximity)*_sparsity)
    print(_sparsity)
    print(_proximity)



test_adult_income_data()
