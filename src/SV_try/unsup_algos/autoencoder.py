"""
Autoencoder Binary Anomaly Detector

Unsupervised anomaly detection using PyTorch Autoencoder.
Trains on all data, uses reconstruction error threshold.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class AutoencoderModel(nn.Module):
    """PyTorch Autoencoder for anomaly detection via reconstruction error."""
    
    def __init__(self, input_dim, encoder_layers=[64, 32], latent_dim=16, 
                 decoder_layers=[32, 64], activation='relu', dropout=0.2):
        super(AutoencoderModel, self).__init__()
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()
        
        # Build encoder
        encoder_layers_list = []
        prev_dim = input_dim
        for hidden_dim in encoder_layers:
            encoder_layers_list.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers_list.append(self.activation)
            encoder_layers_list.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Latent layer
        encoder_layers_list.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers_list)
        
        # Build decoder
        decoder_layers_list = []
        prev_dim = latent_dim
        for hidden_dim in decoder_layers:
            decoder_layers_list.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers_list.append(self.activation)
            decoder_layers_list.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers_list.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers_list)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class BinaryAutoencoderDetector:
    """
    Autoencoder for binary anomaly detection.
    Trains on all data, uses reconstruction error threshold.
    """
    
    def __init__(self, name: str = 'Autoencoder', config: Dict = None, contamination: float = 0.2):
        self.name = name
        self.config = config or {}
        self.ae = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.contamination = contamination
        
    def fit(self, X_train, y_train=None):
        """Train autoencoder on all data, set threshold on normal samples."""
        X_scaled = self.scaler.fit_transform(X_train)
        input_dim = X_scaled.shape[1]
        
        # Build autoencoder
        self.ae = AutoencoderModel(
            input_dim=input_dim,
            encoder_layers=self.config.get('encoder_layers', [64, 32]),
            latent_dim=self.config.get('latent_dim', 16),
            decoder_layers=self.config.get('decoder_layers', [32, 64]),
            activation=self.config.get('activation', 'relu'),
            dropout=self.config.get('dropout', 0.2)
        ).to(self.device)
        
        # Training parameters
        epochs = self.config.get('epochs', 50)
        batch_size = self.config.get('batch_size', 256)
        learning_rate = self.config.get('learning_rate', 0.001)
        patience = self.config.get('early_stopping_patience', 10)
        
        # Use only normal samples for training if labels provided
        if y_train is not None:
            normal_mask = y_train == 0
            X_normal = X_scaled[normal_mask]
            if len(X_normal) < 100:
                X_normal = X_scaled
        else:
            X_normal = X_scaled
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_normal).to(self.device)
        
        # Train/validation split (80/20)
        n_train = int(len(X_normal) * 0.8)
        X_train_tensor = X_tensor[:n_train]
        X_val_tensor = X_tensor[n_train:] if n_train < len(X_normal) else X_tensor
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.ae.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.ae.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.ae(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.ae.eval()
            with torch.no_grad():
                val_output = self.ae(X_val_tensor)
                val_loss = criterion(val_output, X_val_tensor).item()
            self.ae.train()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        self.ae.eval()
        
        # Compute threshold based on normal data reconstruction error
        with torch.no_grad():
            X_normal_tensor = torch.FloatTensor(X_normal).to(self.device)
            reconstructed = self.ae(X_normal_tensor)
            errors = torch.mean((X_normal_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        # Set threshold based on contamination rate
        threshold_percentile = (1 - self.contamination) * 100
        self.threshold = np.percentile(errors, threshold_percentile)
        
        return self
    
    def predict(self, X):
        """Predict: 1 = anomaly (attack), 0 = normal."""
        errors = self._compute_reconstruction_error(X)
        return (errors > self.threshold).astype(int)
    
    def predict_scores(self, X):
        """Return reconstruction errors as anomaly scores."""
        return self._compute_reconstruction_error(X)
    
    def _compute_reconstruction_error(self, X):
        """Compute per-sample reconstruction error."""
        X_scaled = self.scaler.transform(X)
        self.ae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed = self.ae(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        return errors


def model(X_train, y_train, X_val=None, y_val=None, contamination=0.2):
    """
    Create and train an Autoencoder detector.
    
    Args:
        X_train: Training features
        y_train: Training labels (0=normal, 1=attack)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        contamination: Expected proportion of anomalies
    
    Returns:
        Trained BinaryAutoencoderDetector
    """
    print("Training Autoencoder...")
    
    config = {
        'encoder_layers': [64, 32],
        'latent_dim': 16,
        'decoder_layers': [32, 64],
        'epochs': 50,
        'batch_size': 256,
        'learning_rate': 0.001,
        'early_stopping_patience': 10
    }
    
    detector = BinaryAutoencoderDetector(
        name='Autoencoder',
        config=config,
        contamination=contamination
    )
    
    detector.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = detector.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        print(f"Autoencoder Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return detector
