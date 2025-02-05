import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

def train_model(input_dir="../data/processed_data", model_dir="."):
    # Charger les données
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv").values.ravel()
    
    # Charger les meilleurs paramètres trouvés avec GridSearch
    with open(f"{model_dir}/best_model_params.pkl", "rb") as f:
        best_params = pickle.load(f)
    
    # Initialiser et entraîner le modèle avec les meilleurs paramètres
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    # Sauvegarder le modèle entraîné
    with open(f"{model_dir}/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Modèle entraîné et sauvegardé avec succès !")

if __name__ == "__main__":
    train_model()

