# src/qml/predict_with_qml.py

import torch
from src.qml.qml_model import HybridQMLModel
from src import config
import os

if __name__ == "__main__":
    print("--- Phase 2.3: Prediction with Trained QML Model ---")

    if not os.path.exists(config.QML_MODEL_PATH):
        print("Error: Model file not found. Please train the model first by running:")
        print("python src/qml/train_qml_model.py")
    else:
        # 1. Load the trained model
        model = HybridQMLModel()
        model.load_state_dict(torch.load(config.QML_MODEL_PATH))
        model.eval()
        print("Trained QML model loaded successfully.")

        # 2. Create new data point for prediction
        new_data = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
        # Expected value: sin(0.5*pi) * cos(-0.2*pi) = 1.0 * 0.8090 = 0.8090
        expected_y = torch.sin(torch.pi * new_data[0,0]) * torch.cos(torch.pi * new_data[0,1])


        # 3. Make a prediction
        with torch.no_grad():
            prediction = model(new_data)

        print(f"\nInput data: {new_data.numpy()[0]}")
        print(f"Expected output: {expected_y.item():.4f}")
        print(f"Model prediction: {prediction.item():.4f}")