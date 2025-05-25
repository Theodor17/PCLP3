import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = train.drop(columns = ['Greutate (Kg)', 'Pericol de disparitie'])
y_train_for_regression = train['Greutate (Kg)']

x_test = test.drop(columns=['Greutate (Kg)', 'Pericol de disparitie'])
y_test_for_regression = test['Greutate (Kg)']

cat_cols = ['Nume Specie', 'Categorie', 'Habitat', 'Culoare']
num_cols = ['Durata de viata (ani)', 'Viteza (Km/h)', 'Inaltime (m)']

# preprocesor care face OneHotEncoding pentru variabilele categorice si pastrarea coloanelor numerice
preprocessor = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(drop = 'first'), cat_cols)
    ], 
    remainder = 'passthrough'  
)

# modelele pentru regresie
models_regression = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(),
    'SVR': SVR()
}

# evaluarea regresiei
def evaluate_regression_model(model, x_test, y_test):

    y_predicted = model.predict(x_test)
    # calculam RMSE, MAE, R^2
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    mae = mean_absolute_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    return rmse, mae, r2

# compararea modelelor de regresie
results_regression = []

for name, model in models_regression.items():

    # pipeline cu preprocesor si modelul de regresie
    pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(x_train, y_train_for_regression)  # antrenare
    
    # evaluarea modelului
    rmse, mae, r2 = evaluate_regression_model(pipeline, x_test, y_test_for_regression)
    
    # Stocam rezultatele intr-un tabel comparativ
    results_regression.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2
    })

# tabel comparativ pentru regresie
results = pd.DataFrame(results_regression)
print("\nTabel comparativ regresie:\n", results)
print("\n")

x_train = train.drop(columns = ['Greutate (Kg)', 'Pericol de disparitie'])
y_train = train['Pericol de disparitie'].map({'Nu': 0, 'Da': 1})

x_test = test.drop(columns = ['Greutate (Kg)', 'Pericol de disparitie'])
y_test = test['Pericol de disparitie'].map({'Nu': 0, 'Da': 1})

# preprocesor pentru encoding
preprocessor = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(drop = 'first'), cat_cols)
    ],
    remainder = 'passthrough'
)

models_classification = {
    'Logistic Regression': LogisticRegression(max_iter = 1024),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier()
}

# evaluare clasificare
def evaluate_classification_model(model, x_test, y_test):

    # evaluez acuratetea, f1 score si rou auc
    y_predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    return accuracy, f1, roc_auc

# lista pentru rezultate
results = []

# antrenare si evaluare modele
for name, model in models_classification.items():

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(x_train, y_train)
    accuracy, f1, roc_auc = evaluate_classification_model(pipeline, x_test, y_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    })

# tabel cu comparatii
rezultate = pd.DataFrame(results)
print("\nTabel comparativ clasificare:\n", rezultate)
print("\n")