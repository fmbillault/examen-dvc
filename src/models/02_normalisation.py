import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(input_dir="../data/processed_data", output_dir="../data/processed_data"):
    # Charger les données d'entraînement et de test
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    # Sélectionner uniquement les colonnes numériques
    numeric_columns = X_train.select_dtypes(include=["number"]).columns
    X_train_numeric = X_train[numeric_columns]
    X_test_numeric = X_test[numeric_columns]

    # Initialiser le scaler et l'ajuster sur l'ensemble d'entraînement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)  # Transformer X_test avec les mêmes paramètres

    # Convertir en DataFrame en conservant les noms des colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_columns)

    # Sauvegarder les nouveaux datasets
    X_train_scaled.to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{output_dir}/X_test_scaled.csv", index=False)

    print("Données numériques normalisées et sauvegardées avec succès !")

if __name__ == "__main__":
    normalize_data()

