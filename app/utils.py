import torch
import joblib
import numpy as np
from collections import deque
from .model import TrajectoryTransformer
import logging
import warnings

# Suppress specific warnings related to the scaler
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHandler:
    """
    A class to manage the model loading, preprocessing, and prediction process.
    This includes handling the speed data, buffering it, and triggering predictions
    when the buffer exceeds a defined threshold.

    Attributes:
        device (torch.device): The device for model inference (CPU or GPU).
        window_size (int): The size of the input window (number of speeds to consider for prediction).
        buffer_threshold (int): The minimum number of speeds required to trigger a prediction.
        label_encoder (LabelEncoder): The label encoder used to transform predicted labels.
        scaler (StandardScaler): The scaler used to normalize input data.
        model (TrajectoryTransformer): The trained model.
        buffer (deque): A buffer holding the most recent speed values.
    """

    def __init__(self, model_path, label_encoder_path, scaler_path, device='cpu', window_size=10, buffer_threshold=5):
        """
        Initializes the ModelHandler class.

        Args:
            model_path (str): Path to the saved model (.pth file).
            label_encoder_path (str): Path to the saved LabelEncoder (.joblib file).
            scaler_path (str): Path to the saved StandardScaler (.joblib file).
            device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
            window_size (int, optional): Size of the input window. Defaults to 10.
            buffer_threshold (int, optional): Minimum buffer size to trigger prediction. Defaults to 5.
        """
        self.device = torch.device(device)
        self.window_size = window_size  # Number of speeds to consider for prediction
        self.buffer_threshold = buffer_threshold  # Trigger prediction once buffer exceeds this threshold
        self.label_encoder = self.load_label_encoder(label_encoder_path)
        self.scaler = self.load_scaler(scaler_path)
        self.model = self.load_model(model_path)
        self.buffer = deque(maxlen=window_size)  # Store the most recent speeds (up to window_size)

    def load_model(self, model_path):
        """
        Loads the trained model from a file.

        Args:
            model_path (str): Path to the saved model.

        Returns:
            TrajectoryTransformer: The loaded model.
        """
        feature_size = 1  # Only speed data is used
        num_classes = len(self.label_encoder.classes_)

        model = TrajectoryTransformer(
            feature_size=feature_size,
            num_classes=num_classes,
            d_model=128,        # Must match training configuration
            nhead=8,            # Must match training configuration
            num_layers=4,       # Must match training configuration
            window_size=self.window_size
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def load_label_encoder(self, label_encoder_path):
        """
        Loads the LabelEncoder from a file.

        Args:
            label_encoder_path (str): Path to the saved LabelEncoder.

        Returns:
            LabelEncoder: The loaded LabelEncoder.
        """
        return joblib.load(label_encoder_path)

    def load_scaler(self, scaler_path):
        """
        Loads the StandardScaler from a file.

        Args:
            scaler_path (str): Path to the saved StandardScaler.

        Returns:
            StandardScaler: The loaded StandardScaler.
        """
        return joblib.load(scaler_path)

    def process_speeds_and_check_prediction(self, speeds):
        """
        Processes a list of speed values, manages the buffer, and checks if a prediction
        should be triggered. Prediction is triggered if the buffer contains enough data
        (exceeds the buffer threshold).

        Args:
            speeds (list): A list of speed values.

        Returns:
            tuple: (window_tensor, position_ids, mask_tensor) if prediction should be made, else None.
        """
        # Add all speeds to the buffer
        for speed in speeds:
            self.buffer.append(speed)

        # If the buffer exceeds the threshold, proceed with prediction
        if len(self.buffer) >= self.buffer_threshold:
            logger.info(f"Stored speeds in buffer: {list(self.buffer)}")

            # Prepare the input window
            padded_window = list(self.buffer)
            if len(padded_window) < self.window_size:
                # Pad the window with zeros if it's smaller than the window size
                padding_length = self.window_size - len(padded_window)
                padded_window = [0.0] * padding_length + padded_window
                mask = [True] * padding_length + [False] * len(self.buffer)
            else:
                mask = [False] * self.window_size

            # Normalize the window and convert to tensors
            window = np.array(padded_window, dtype=np.float32).reshape(-1, 1)
            window = self.scaler.transform(window)
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            position_ids = torch.arange(self.window_size).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            # Clear the buffer after prediction
            self.buffer.clear()
            return window_tensor, position_ids, mask_tensor

        # Return None if not enough data is available for prediction
        self.buffer.clear()
        return None
