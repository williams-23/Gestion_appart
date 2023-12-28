from matplotlib import pyplot as plt
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)
# Charger les données depuis le fichier CSV
donnees = pd.read_csv("Classeur2.csv")


X = donnees.drop(['agglomeration'], axis=1)
y = donnees["agglomeration"]
X = donnees.drop(['Observatory'], axis=1)
y = donnees["Observatory"]

donnees.head()

# Vérifier la taille des données après le nettoyage
if donnees.shape[0] == 0:
    raise ValueError("Toutes les données ont été supprimées après le nettoyage. Vérifiez votre processus de nettoyage.")

# Spécifier les colonnes à utiliser pour la prédiction
colonnes_a_utiliser = ['loyer_moyen', 'surface_moyenne', 'nombre_logements']
X = donnees[colonnes_a_utiliser]
y = donnees['moyenne_loyer_mensuel']  # Remplacez par le nom de votre colonne cible

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = X_train.join(y_train)

train_data

train_data.hist(figsize=(15, 8))

from sklearn.impute import SimpleImputer

# Créer un objet imputer
imputer = SimpleImputer(strategy='mean')  # Vous pouvez utiliser 'median' ou 'most_frequent' selon votre choix
target_imputer = SimpleImputer(strategy='mean')

# Appliquer l'imputation sur X_train
X_train_imputed = imputer.fit_transform(X_train)

# Appliquer la même imputation sur X_test
X_test_imputed = imputer.transform(X_test)

# Appliquer l'imputation sur y_train
y_train_imputed = target_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Appliquer la même imputation sur y_test
y_test_imputed = target_imputer.transform(y_test.values.reshape(-1, 1)).ravel()

# Initialiser le modèle de régression linéaire
model = LinearRegression()

model.fit(X_train_imputed, y_train_imputed)
predictions = model.predict(X_test_imputed)

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(X_test_imputed)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        loyer_moyen = float(request.form['loyer_moyen'])
        nombre_logements = float(request.form['nombre_logements'])
        surface_moyenne = float(request.form['surface_moyenne'])
        #agglomeration = input("Agglomération (urbain/suburbain/rural) : ")

        # Créer un DataFrame avec les informations de l'utilisateur
        nouvelles_donnees = pd.DataFrame({
            'loyer_moyen': [loyer_moyen],
            'nombre_logements': [nombre_logements],
            'surface_moyenne': [surface_moyenne],
            #'Agglomeration': [agglomeration]
        })


        # Faire la prédiction
        prediction = model.predict(nouvelles_donnees)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
