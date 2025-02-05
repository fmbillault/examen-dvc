import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_file="../data/raw_data/raw.csv", output_dir="../data/processed_data", test_size=0.2, random_state=42):
    # Charger les données
    df = pd.read_csv(input_file)
    
    # Séparer les features et la variable cible
    X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1]   # Dernière colonne (silica_concentrate)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Sauvegarder les datasets
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print("Données divisées et sauvegardées avec succès !")

if __name__ == "__main__":
    split_data()

