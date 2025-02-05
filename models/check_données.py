import pandas as pd

# Charger le fichier CSV (remplace 'data.csv' par ton fichier)
file_path = "raw.csv"
df = pd.read_csv(file_path)

# Afficher les premières lignes
print("\n📌 Aperçu des données :")
print(df.head())

# Afficher les statistiques générales
print("\n📊 Statistiques descriptives :")
print(df.describe(include="all"))  # inclut aussi les colonnes catégorielles

# Vérifier les valeurs manquantes
print("\n🔍 Valeurs manquantes :")
print(df.isnull().sum())

# Afficher les types de données
print("\n🧐 Types de données :")
print(df.dtypes)

