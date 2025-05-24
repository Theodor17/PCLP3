import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# greutate - este variabila tinta pentru regresie, adica ce vrem sa prezicem
# pericol de disparitie este o alta variabila tinta pentru clasificare, dar pentru regresie o eliminam si pe ea
x_train = train.drop(columns = ['Greutate (Kg)', 'Pericol de disparitie'])
y_train_for_regression = train['Greutate (Kg)']

x_test = test.drop(columns = ['Greutate (Kg)', 'Pericol de disparitie'])
y_test_for_regression = test['Greutate (Kg)']

# codificarea variabilelor categorice cu OneHotEncoder
cat_cols = ['Nume Specie', 'Categorie', 'Habitat', 'Culoare']

# cream un preprocesor care aplica OneHotEncoding pe coloanele categorice
preprocessor = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(drop = 'first'), cat_cols)  
    ], 
    remainder='passthrough'  # pastram coloanele numerice asa cum sunt
)

# definim cele 3 modele de regresie
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(),
    'SVR': SVR()
}

# antrenarea si evaluarea modelelor
for name, model in models.items():
    print(f"\nAntrenare si evaluare model: {name}")

    # pipeline cu preprocesorul si modelele de regresie
    pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # antrenam modelul
    pipeline.fit(x_train, y_train_for_regression)

    # predictii pe setul de testare
    y_predictions = pipeline.predict(x_test)

    # evaluare
    r2 = r2_score(y_test_for_regression, y_predictions)
    rmse = np.sqrt(mean_squared_error(y_test_for_regression, y_predictions))  # calculam

    print(f"{name} - R^2: {r2}")
    print(f"{name} - RMSE: {rmse}\n")
