from discern.discern_tabular import DisCERNTabular
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_cancer_risk():
    data_df = pd.read_csv('lung_cancer.csv')
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
    x_train= scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Data transform complete!")

    rfx = RandomForestClassifier(n_estimators=500)
    rfx.fit(x_train, y_train)
    print(accuracy_score(y_test, rfx.predict(x_test)))
    print("Training classifier complete!")

    x_test = x_test[:10]
    y_test = rfx.predict(x_test[:10])
    cat_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    sparsity = []
    proximity = []
    discern = DisCERNTabular(rfx, 'LIME', 'Q')
    discern.init_data(x_train, y_train, [c for c in df.columns if c!='Level'], ['Low', 'Medium', 'High'], cat_feature_indices=cat_indices)

    for idx in range(len(x_test)):
        if y_test[idx] == 0:
            continue
        cf, s, p = discern.find_cf(x_test[idx], y_test[idx], desired_class='Low')
        print(s)
        print(p)
        sparsity.append(s)
        proximity.append(p)

    _sparsity = sum(sparsity)/len(sparsity)
    _proximity = sum(proximity)/(len(proximity)*_sparsity)
    print(_sparsity)
    print(_proximity)


test_cancer_risk()