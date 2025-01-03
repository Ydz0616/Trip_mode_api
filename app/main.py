from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from .utils import ModelHandler
import logging
from typing import List
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
app = FastAPI(title="Trajectory Transformer API")

# Initialize the ModelHandler with paths to the trained model, label encoder, and scaler
model_path = 'artifacts/best_model.pth'
label_encoder_path = 'artifacts/label_encoder.joblib'
scaler_path = 'artifacts/scaler.joblib'
device = 'cpu'  # Ensure to use CPU (can be changed to 'cuda' if available)

model_handler = ModelHandler(
    model_path=model_path,
    label_encoder_path=label_encoder_path,
    scaler_path=scaler_path,
    device=device,
    window_size=100,               # Window size (number of speed values to consider)
    buffer_threshold=5           # Buffer threshold to trigger prediction
)

# Define the input schema for prediction requests
class PredictionRequest(BaseModel):
    speeds: List[float]  # List of speed values

# Define the response schema
class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint to make predictions using the trajectory model based on the list of speed values provided.

    Args:
        request (PredictionRequest): A list of speed values for prediction.

    Returns:
        PredictionResponse: The predicted class label and the corresponding probabilities.
    """
    logger.info(f"Received prediction request with speeds: {request.speeds}")
    try:
        # Process the speeds and check if prediction can be made
        preprocess_result = model_handler.process_speeds_and_check_prediction(request.speeds)

        if preprocess_result is None:
            # Not enough data in the buffer to proceed with prediction
            return {"prediction": "Insufficient data", "probabilities": {}}
        else:
            window_tensor, position_ids, mask = preprocess_result

            # Move tensors to the appropriate device (CPU or GPU)
            window_tensor = window_tensor.to(model_handler.device)
            position_ids = position_ids.to(model_handler.device)
            mask = mask.to(model_handler.device)

            # Make prediction using the trained model
            with torch.no_grad():
                outputs = model_handler.model(
                    x=window_tensor,
                    position_ids=position_ids,
                    src_key_padding_mask=mask
                )
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]

            # Decode the label using the label encoder
            predicted_label = model_handler.label_encoder.inverse_transform([predicted_class])[0]

            # Create a dictionary for probabilities
            prob_dict = {label: float(prob) for label, prob in zip(model_handler.label_encoder.classes_, probabilities)}

            logger.info(f"Prediction successful: {predicted_label}")
            return PredictionResponse(
                prediction=predicted_label,
                probabilities=prob_dict
            )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Trajectory Transformer API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
