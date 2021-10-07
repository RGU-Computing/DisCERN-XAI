from discern.discern_tabular import DisCERNTabular
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_df = pd.read_csv('adult_income.csv')
df = data_df.copy()
print("Reading data complete!")
df = df.drop("Unnamed: 0", axis=1)
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

x_test = x_test[0]
y_test = svm.predict([x_test])

discern = DisCERNTabular(svm, 'LIME', 'Q')
discern.init_data(x_train, y_train, [c for c in df.columns if c!='salary'], ['<=50K', '>50K'], cat_feature_indices=[])


cf, _, _ = discern.find_cf(x_test, y_test)

x = scalar.inverse_transform([x_test])
c = scalar.inverse_transform([cf])

cls = list(df.columns)
x=pd.DataFrame(x,columns=[c for c in df.columns if c!='salary'])
c=pd.DataFrame(c,columns=[c for c in df.columns if c!='salary'])
x['salary'] = y_test
c['salary'] = 1 # Temporary

x=x.to_dict(orient="index")[0]
c=c.to_dict(orient="index")[0]

discern.show_cf(x, c)