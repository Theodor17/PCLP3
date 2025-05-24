import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# luam input-ul
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# clasificam coloanele cu date numerice si coloanele cu date nenumerice
num_cols = ['Greutate (Kg)', 'Durata de viata (ani)', 'Viteza (Km/h)', 'Inaltime (m)']
cat_cols = ['Nume Specie', 'Categorie', 'Habitat', 'Culoare']  

# separam setul de date in features si target
x_train = train.drop('Pericol de disparitie', axis = 1)
y_train = train['Pericol de disparitie']

x_test = test.drop('Pericol de disparitie', axis = 1)
y_test = test['Pericol de disparitie']

# variabile numerice => imputare mediana & standardizare 
numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# variabile categorice => imputare moda + one-hot encoding
categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

"""
Este nevoie de normalizarea datelor pentru ca modelul sa invete mult mai usor (dat fiind faptul ca variatia este redusa dupa
normalizare). 
One-hot encoding-ul ajuta la transformarea coloanelor ce nu au date numerice in coloane binare pentru ca modelele sa poata
intelege aceste date si sa le prelucreze mult mai usor.
"""

# combinam cele doua pipeline-uri intr-un singur preprocesor
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# aplicam preprocesarea pe datele de antrenament & test
x_train_after_processing = preprocessor.fit_transform(x_train)
x_test_after_processing = preprocessor.transform(x_test)

# verificam datele pentru primele si ultimele 5 randuri
feature_names_num = num_cols
cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
all_feature_names = np.concatenate([feature_names_num, cat_feature_names])

x_train_df = pd.DataFrame(x_train_after_processing.toarray() if hasattr(x_train_after_processing, "toarray") 
                        else x_train_after_processing, columns = all_feature_names)
print("\n\n", x_train_df)