import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def optimize_model(input_dir="../data/processed_data", output_dir="."):
    # Charger les données
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv").values.ravel()
    
    # Définir le modèle et les paramètres à tester
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Exécuter GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Sauvegarder les meilleurs paramètres
    best_params = grid_search.best_params_
    with open(f"{output_dir}/best_model_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    
    print("Meilleurs paramètres trouvés et sauvegardés !")

if __name__ == "__main__":
    optimize_model()

