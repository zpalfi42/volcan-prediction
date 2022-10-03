import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics

df = pd.read_csv('jm_train.csv')
df_test = pd.read_csv('jm_X_test.csv')

Features = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']]
target = df['target']
F_test = df_test[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']]

clf = RandomForestClassifier(n_estimators=100)
clf.fit(Features, target)

T_pred = clf.predict(F_test)

final = pd.DataFrame(T_pred, columns=['final_status'])
final.to_csv('final.csv', index=False)