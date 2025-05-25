import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# normalizare pentru variabilele numerice
num_cols = ['Greutate (Kg)', 'Durata de viata (ani)', 'Viteza (Km/h)', 'Inaltime (m)']
train[num_cols] = train[num_cols].apply(pd.to_numeric, errors = 'coerce')
test[num_cols] = test[num_cols].apply(pd.to_numeric, errors = 'coerce')

# encodare pentru variabilele categorice
cat_cols = ['Nume Specie', 'Categorie', 'Habitat', 'Culoare']
train_cat = train[cat_cols] 
test_cat = test[cat_cols]   

# inlocuiesc valorile lipsa
imputer_num = SimpleImputer(strategy = 'median')
imputer_cat = SimpleImputer(strategy = 'most_frequent')

train[num_cols] = imputer_num.fit_transform(train[num_cols])
test[num_cols] = imputer_num.transform(test[num_cols])

# folosim la subpunctele urmatoare
# incercam sa obtinem un nume valid de fisier dintr-un nume de coloana
def curata_nume_fisier(nume): 
    return nume.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')

# a) analiza valorilor lipsa
def analiza_valori_lipsa(df, nume = 'dataset'):

    print(f"\nAnaliza valori lipsa pentru {nume}:")
    # calculam totalul valorilor lipsa si in procent
    total = df.isnull().sum()
    procent = (total / len(df)) * 100
    missing_data = pd.DataFrame({'Total lipsa': total, 'Procent (%)': procent})
    print(missing_data[missing_data['Total lipsa'] > 0])
    print("\n")
    
    # vizualizare valori lipsa
    plt.figure(figsize = (10, 4))
    sns.heatmap(df.isnull(), cbar = False, yticklabels = False, cmap = 'viridis')
    plt.title(f"valori lipsa: {nume}")
    
    plt.savefig(f'no_data_for_{nume}.png')
    plt.close()

analiza_valori_lipsa(train, 'train')
analiza_valori_lipsa(test, 'test')

# b) statistici descriptive
def statistici_descriptive(df, nume = 'dataset'):
    
    # selectam coloanele numerice
    numerice = df.select_dtypes(include = ['float64', 'int64'])
    if not numerice.empty:
        print(numerice.describe())
    
    print("\n")
    # acum selectam coloanele cu date categorice
    categorice = df.select_dtypes(include = 'object')
    if not categorice.empty:
        for col in categorice.columns:
            print(f"Coloana '{col}':")
            print(categorice[col].value_counts())  # afisam frecventele pentru fiecare valoare categorica a coloanei
            print("\n")

statistici_descriptive(train, 'train')
statistici_descriptive(test, 'test')

# c) analiza distributiei variabilelor (histograme + grafice)
def histograme_numerice(df, cols, nume = 'dataset'):
    
    # for a better org
    if not os.path.exists('eda_plots'):
        os.makedirs('eda_plots')  

    for col in cols:
        # ca sa pot avea nume valide de fisier
        nume_valid = curata_nume_fisier(col)
        
        # vom crea histograme pt fiecare coloana numerica
        plt.figure(figsize = (8, 6))
        sns.histplot(df[col], kde = True, bins = 30)  # histograma cu curba de densitate
        plt.title(f'Histograma pentru {col}')
        plt.xlabel(col)
        plt.ylabel('Frecventa')

        # salveaza fisierul in format .png in /eda_plots
        plt.savefig(f'eda_plots/histogram_{nume_valid}.png')
        plt.close()  # evitam suprapunerea graficelor prin inchiderea fisierelor

# countplot pentru variabilele categorice
def countplot_categorice(df, cols, nume = 'dataset'):

    # for a better org
    if not os.path.exists('eda_plots'):
        os.makedirs('eda_plots') 

    for col in cols:
        # ca sa pot avea nume valide de fisier
        nume_valid = curata_nume_fisier(col)

        # vom crea countplots pt fiecare coloana categorica
        plt.figure(figsize = (8, 6))
        sns.countplot(x = col, data = df, order = df[col].value_counts().index)  
        plt.title(f'Countplot pentru {col}')
        plt.xlabel(col)
        plt.ylabel('Nr aparitii')

        # salveaza fisierul in format .png in /eda_plots
        plt.savefig(f'eda_plots/histogram_{nume_valid}.png')
        plt.close()  # evitam suprapunerea graficelor prin inchiderea fisierelor

histograme_numerice(train, num_cols, 'train')
countplot_categorice(train, cat_cols, 'train')

# d) detectarea outlierilor cu boxplot & IQR 
def detectare_outlieri(df, cols):

    # for a better org
    if not os.path.exists('outliers'):
        os.makedirs('outliers') 

    for col in cols:
        nume_valid = curata_nume_fisier(col)
        
        # creeaza boxplot
        plt.figure(figsize = (6,4))
        sns.boxplot(x = df[col])
        plt.title(f'Boxplot pentru {nume_valid}')
        plt.savefig(f'outliers/boxplot_{nume_valid}.png')
        plt.close() 

        # IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # identificare outliers
        outlieri = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        
        print(f"Outlieri pe coloana '{col}': {len(outlieri)}")
        print(outlieri[[col]])
        print("\n")

detectare_outlieri(train, num_cols)

# e) Analiza corelațiilor
plt.figure(figsize = (12, 6))
sns.heatmap(train[num_cols].corr(), annot = True, cmap = 'coolwarm', center = 0)
plt.title('Matrice de corelatii')
plt.savefig('corelatii_train.png')
plt.close()

# f) analiza relatiilor cu variabila tinta (pericol de disparitie)
train['Pericol_numeric'] = train['Pericol de disparitie'].map({'Nu':0, 'Da':1})

# for a better org
if not os.path.exists('analiza_relationala'):
    os.makedirs('analiza_relationala')  

# grafice violin pentru variabilele numerice vs Pericol de disparitie
for col in num_cols:
    plt.figure(figsize = (6, 4))
    sns.violinplot(x = 'Pericol_numeric', y = col, data = train)
    plt.title(f'{col} vs Pericol de disparitie')
    plt.xlabel('Pericol de disparitie (0 = Nu, 1 = Da)')
    plt.ylabel(col)

    plt.savefig(f'analiza_relationala/violin_{curata_nume_fisier(col)}_pericol.png')
    plt.close()  

# grafice de tip countplot pentru variabilele categorice vs Pericol de disparitie
for col in cat_cols:
    plt.figure(figsize = (8, 4))
    sns.countplot(x = col, hue = 'Pericol de disparitie', data = train)
    plt.title(f'{col} vs Pericol de disparitie')
    
    plt.savefig(f'analiza_relationala/countplot_{curata_nume_fisier(col)}_pericol.png')
    plt.close()  