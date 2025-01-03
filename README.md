# Trajectory Transformer API

This API serves a trajectory prediction model using speed data as input. It leverages a pre-trained model to predict the class label and associated probabilities based on a series of speed values.

## Requirements

- FastAPI
- Uvicorn
- PyTorch
- Pydantic
- Scikit-learn

## Setup

Ensure the following model files are available in the `artifacts` directory:

- `best_model.pth` (Pre-trained trajectory model)
- `label_encoder.joblib` (Label encoder for decoding predictions)
- `scaler.joblib` (Scaler for input data normalization)

## API Endpoints

### POST /predict

Predicts the class label and probabilities based on a list of speed values.
