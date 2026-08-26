import pandas as pd
import numpy as np
import ast
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('Datasets/bangalore-police-traffic-violation-dataset-2023.csv')
df.info()

df['created_datetime'] = pd.to_datetime(df['created_datetime'],errors='coerce')
df = df.dropna(subset=['created_datetime','violation_type'])
df["hour"] = df['created_datetime'].dt.hour
df["day_of_week"] = df['created_datetime'].dt.dayofweek
df['month'] = df['created_datetime'].dt.month
df['is_weekend'] = (df['day_of_week'] >=5).astype(int)
df['peak_hour'] = (df['hour'].between(7,10) | df['hour'].between(17,22)).astype(int)


#the problem -> from line 24 to 33
def normalise_violations(x):
    v = ast.literal_eval(x)
    return str(sorted(v))

df["violation_type"] = df["violation_type"].apply(normalise_violations)
df['violation_list'] = df['violation_type'].apply(ast.literal_eval)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['violation_list'])
print(mlb.classes_)

num_features = ["latitude","longitude","center_code","hour","day_of_week","month","is_weekend","peak_hour"]
cat_features = ["location","vehicle_type","police_station","junction_name"]
features = (num_features + cat_features)

x = df[features].copy()

for c in num_features:
    x[c] = x[c].fillna(x[c].median())

for c in cat_features:
    x[c] = x[c].fillna("Unknown")

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

models = {}
predictions = np.zeros_like(y_test)
for i , v in enumerate(mlb.classes_):
    print(f"Training catboost for : {v}")
    model = CatBoostClassifier(iterations=500,depth=6,learning_rate=0.1,verbose=50,loss_function="Logloss",eval_metric="Accuracy",random_seed=42)
    model.fit(x_train,y_train[:,1],cat_features=cat_features)
    predictions[:,1] = model.predict(x_test).flatten()
    models[v] = model
    
predictions = predictions.astype(int)
print("MULTILABEL CATBOOST RESULTS : ")
print(f"Subset Accuracy : {accuracy_score(y_test,predictions)}")
print(classification_report(y_test,predictions,target_names=mlb.classes_,zero_division=0))