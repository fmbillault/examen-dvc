import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(input_dir="../data/processed_data", model_dir=".", output_dir="../metrics"):
    # Charger les données de test
    X_test = pd.read_csv(f"{input_dir}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{input_dir}/y_test.csv").values.ravel()
    
    # Charger le modèle entraîné
    with open(f"{model_dir}/trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Faire des prédictions
    y_pred = model.predict(X_test)
    
    # Évaluer les performances du modèle
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Sauvegarder les scores dans un fichier JSON
    scores = {"Mean Squared Error": mse, "R2 Score": r2}
    with open(f"{output_dir}/scores.json", "w") as f:
        json.dump(scores, f, indent=4)
    
    # Sauvegarder les prédictions dans un fichier CSV
    predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    predictions.to_csv(f"../data/predictions.csv", index=False)
    
    print("Évaluation terminée. Scores et prédictions sauvegardés !")

if __name__ == "__main__":
    evaluate_model()

