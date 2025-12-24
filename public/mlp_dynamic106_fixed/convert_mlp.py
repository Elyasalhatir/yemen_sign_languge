import joblib
import pickle
import json
import numpy as np
import pathlib

# Paths
MODEL_DIR = pathlib.Path(__file__).parent
MODEL_PATH = MODEL_DIR / "mlp_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
INV_MAP_PATH = MODEL_DIR / "idx_to_word.pkl"
OUTPUT_PATH = MODEL_DIR / "mlp_model.json"

def convert_model():
    print(f"Loading model from {MODEL_DIR}...")
    
    # Load assets
    try:
        mlp = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        idx_to_word = pickle.load(open(INV_MAP_PATH, "rb"))
        
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model files: {e}")
        return

    # Extract Scaler
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    n_features = scaler.n_features_in_
    print(f"Scaler features: {n_features}")

    # Extract MLP weights
    # MLPClassifier stores weights in coefs_ and intercepts_
    # coefs_ is a list of weight matrices: [Input->H1, H1->H2, ..., Hn->Output]
    # intercepts_ is a list of bias vectors: [H1, H2, ..., Output]
    
    weights = []
    biases = []
    
    for w in mlp.coefs_:
        weights.append(w.tolist())
    
    for b in mlp.intercepts_:
        biases.append(b.tolist())
        
    layers = [w.shape for w in mlp.coefs_]
    print(f"MLP Layers (Weights): {layers}")

    # Create export dictionary
    model_data = {
        "n_features": n_features,
        "scaler": {
            "mean": scaler_mean,
            "scale": scaler_scale
        },
        "layers": {
            "weights": weights,
            "biases": biases,
            "activations": ["relu"] * (len(weights) - 1) + ["softmax"] # Assuming standard MLP structure from sk-learn default (relu hidden) + softmax for predict_proba
        },
        "classes": idx_to_word
    }
    
    # Save to JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(model_data, f)
        
    print(f"Model converted and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    convert_model()
