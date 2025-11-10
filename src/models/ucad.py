"""
UCAD: Unsupervised Cyberattack Detection
Complete PyTorch implementation with joint training framework
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
import time


# ============================================================
# 1. DAE COMPONENTS
# ============================================================

class DAEEncoder(nn.Module):
    """
    Denoising Autoencoder Encoder.
    Maps input features to latent space representation.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.1):
        super(DAEEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class DAEDecoder(nn.Module):
    """
    Denoising Autoencoder Decoder.
    Reconstructs input from latent space representation.
    """
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(DAEDecoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer to reconstruct input
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


# ============================================================
# 2. PSEUDO-CLASSIFIER (TRANSFORMER-BASED)
# ============================================================

class PseudoClassifier(nn.Module):
    """
    Transformer-based pseudo-classifier.
    Takes latent representations and outputs class probabilities.
    """
    def __init__(
        self, 
        latent_dim: int, 
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super(PseudoClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # LogSoftmax for NLL loss
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, z):
        # z shape: [batch_size, latent_dim]
        # Transformer expects [batch_size, seq_len, features]
        # We treat each sample as a sequence of length 1
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        
        # Pass through transformer
        transformer_out = self.transformer(z_expanded)  # [batch_size, 1, latent_dim]
        
        # Take the output (squeeze sequence dimension)
        transformer_out = transformer_out.squeeze(1)  # [batch_size, latent_dim]
        
        # Classification
        logits = self.classifier(transformer_out)  # [batch_size, num_classes]
        log_probs = self.log_softmax(logits)
        
        return log_probs


# ============================================================
# 3. PSEUDO-LABEL GENERATOR
# ============================================================

class PseudoLabelGenerator:
    """
    Dynamic pseudo-label generator.
    Generates pseudo-labels based on reconstruction error and contamination rate.
    """
    def __init__(self, contamination_rate: float = 0.076):
        self.contamination_rate = contamination_rate
    
    def generate_labels(
        self, 
        X: torch.Tensor, 
        X_recon: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Generate pseudo-labels based on reconstruction error.
        
        Args:
            X: Original input [n_samples, n_features]
            X_recon: Reconstructed input [n_samples, n_features]
        
        Returns:
            Z_pseudo: Latent representations with pseudo-labels [n_samples, latent_dim]
            Y_pseudo: Pseudo-labels [n_samples]
            reconstruction_errors: Array of reconstruction errors
        """
        # Calculate reconstruction error (MSE per sample)
        reconstruction_errors = torch.mean((X - X_recon) ** 2, dim=1).detach().cpu().numpy()
        
        # Sort by reconstruction error (ascending)
        sorted_indices = np.argsort(reconstruction_errors)
        
        # Split point based on contamination rate
        n_samples = len(reconstruction_errors)
        split_point = int(n_samples * (1 - self.contamination_rate))
        
        # Generate pseudo-labels
        pseudo_labels = np.zeros(n_samples, dtype=np.int64)
        pseudo_labels[sorted_indices[split_point:]] = 1  # Top errors = attacks
        
        return torch.from_numpy(pseudo_labels), reconstruction_errors


# ============================================================
# 4. MAIN UCAD DETECTOR
# ============================================================

class UCADDetector:
    """
    Complete UCAD detector with joint training framework.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims_encoder: List[int] = [64, 128],
        hidden_dims_decoder: List[int] = [128, 64],
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dim_feedforward: int = 128,
        dropout: float = 0.1,
        contamination_rate: float = 0.076,
        device: str = 'cpu'
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.contamination_rate = contamination_rate
        self.device = torch.device(device)
        
        # Initialize components
        self.encoder = DAEEncoder(
            input_dim, hidden_dims_encoder, latent_dim, dropout
        ).to(self.device)
        
        self.decoder = DAEDecoder(
            latent_dim, hidden_dims_decoder, input_dim, dropout
        ).to(self.device)
        
        self.classifier = PseudoClassifier(
            latent_dim, 
            num_classes=2,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout
        ).to(self.device)
        
        self.pseudo_label_generator = PseudoLabelGenerator(contamination_rate)
        
        # Training history
        self.history = {
            'recon_loss': [],
            'cls_loss': [],
            'epoch_time': []
        }
    
    def train_ucad(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        lr_dae: float = 1e-3,
        lr_classifier: float = 1e-3,
        patience: int = 7,
        verbose: bool = True
    ):
        """
        Train UCAD using the joint training framework.
        
        Args:
            X_train: Training data [n_samples, n_features]
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            lr_dae: Learning rate for DAE
            lr_classifier: Learning rate for classifier
            patience: Early stopping patience
            verbose: Print training progress
        """
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        
        # Create data loader for reconstruction stage
        train_dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizers
        optimizer_dae = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr_dae
        )
        optimizer_cls = optim.Adam(self.classifier.parameters(), lr=lr_classifier)
        
        # Loss functions
        criterion_recon = nn.MSELoss()
        criterion_cls = nn.NLLLoss()
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            print("\n" + "="*70)
            print("TRAINING UCAD - JOINT END-TO-END FRAMEWORK")
            print("="*70)
            print(f"Training samples: {len(X_train)}")
            print(f"Contamination rate: {self.contamination_rate:.1%}")
            print(f"Expected normal: {int(len(X_train) * (1 - self.contamination_rate))}")
            print(f"Expected attacks: {int(len(X_train) * self.contamination_rate)}")
            print("="*70 + "\n")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # =============================================
            # STAGE 1: RECONSTRUCTION
            # =============================================
            self.encoder.train()
            self.decoder.train()
            
            recon_loss_epoch = 0.0
            for batch in train_loader:
                X_batch = batch[0]
                
                optimizer_dae.zero_grad()
                
                # Forward pass
                Z = self.encoder(X_batch)
                X_recon = self.decoder(Z)
                
                # Reconstruction loss
                loss_recon = criterion_recon(X_recon, X_batch)
                
                # Backward pass
                loss_recon.backward()
                optimizer_dae.step()
                
                recon_loss_epoch += loss_recon.item() * X_batch.size(0)
            
            avg_recon_loss = recon_loss_epoch / len(X_train)
            self.history['recon_loss'].append(avg_recon_loss)
            
            # =============================================
            # STAGE 2: PSEUDO-LABEL GENERATION
            # =============================================
            self.encoder.eval()
            self.decoder.eval()
            
            with torch.no_grad():
                Z_all = self.encoder(X_tensor)
                X_recon_all = self.decoder(Z_all)
            
            # Generate pseudo-labels
            Y_pseudo, recon_errors = self.pseudo_label_generator.generate_labels(
                X_tensor.cpu(), X_recon_all.cpu()
            )
            
            # =============================================
            # STAGE 3: PSEUDO-CLASSIFICATION
            # =============================================
            self.classifier.train()
            
            # Create data loader for pseudo-labeled data
            pseudo_dataset = TensorDataset(Z_all.cpu(), Y_pseudo)
            pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)
            
            cls_loss_epoch = 0.0
            for Z_batch, Y_batch in pseudo_loader:
                Z_batch = Z_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                optimizer_cls.zero_grad()
                
                # Forward pass
                log_probs = self.classifier(Z_batch)
                
                # Classification loss
                loss_cls = criterion_cls(log_probs, Y_batch)
                
                # Backward pass
                loss_cls.backward()
                optimizer_cls.step()
                
                cls_loss_epoch += loss_cls.item() * Z_batch.size(0)
            
            avg_cls_loss = cls_loss_epoch / len(X_train)
            self.history['cls_loss'].append(avg_cls_loss)
            
            # =============================================
            # EPOCH SUMMARY
            # =============================================
            epoch_time = time.time() - epoch_start
            self.history['epoch_time'].append(epoch_time)
            
            if verbose:
                pseudo_dist = np.bincount(Y_pseudo.numpy())
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Recon Loss: {avg_recon_loss:.6f} | "
                      f"Cls Loss: {avg_cls_loss:.6f} | "
                      f"Time: {epoch_time:.1f}s")
                print(f"  Pseudo-labels: Normal={pseudo_dist[0]}, Attack={pseudo_dist[1]}")
            
            # Early stopping
            combined_loss = avg_recon_loss + avg_cls_loss
            if combined_loss < best_loss:
                best_loss = combined_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        if verbose:
            print("\n" + "="*70)
            print("TRAINING COMPLETED")
            print("="*70)
            total_time = sum(self.history['epoch_time'])
            print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"Average epoch time: {np.mean(self.history['epoch_time']):.1f}s")
            print("="*70 + "\n")
    
    def detect(self, X: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform detection on new data.
        
        Args:
            X: Input data [n_samples, n_features]
            batch_size: Batch size for inference
        
        Returns:
            predictions: Binary predictions (0=normal, 1=attack)
            log_probs: Log probabilities for each class [n_samples, 2]
        """
        self.encoder.eval()
        self.classifier.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        all_predictions = []
        all_log_probs = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(X), batch_size):
                X_batch = X_tensor[i:i+batch_size]
                
                # Encode
                Z = self.encoder(X_batch)
                
                # Classify
                log_probs = self.classifier(Z)
                predictions = torch.argmax(log_probs, dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_log_probs.append(log_probs.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        log_probs = np.concatenate(all_log_probs)
        
        return predictions, log_probs
    
    def get_latent_representations(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Get latent space representations for visualization.
        
        Args:
            X: Input data [n_samples, n_features]
            batch_size: Batch size for processing
        
        Returns:
            Z: Latent representations [n_samples, latent_dim]
        """
        self.encoder.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        all_latents = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X_tensor[i:i+batch_size]
                Z = self.encoder(X_batch)
                all_latents.append(Z.cpu().numpy())
        
        return np.concatenate(all_latents)
    
    def get_reconstruction_errors(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Calculate reconstruction errors for analysis.
        
        Args:
            X: Input data [n_samples, n_features]
            batch_size: Batch size for processing
        
        Returns:
            errors: Reconstruction errors [n_samples]
        """
        self.encoder.eval()
        self.decoder.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        all_errors = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X_tensor[i:i+batch_size]
                
                Z = self.encoder(X_batch)
                X_recon = self.decoder(Z)
                
                errors = torch.mean((X_batch - X_recon) ** 2, dim=1)
                all_errors.append(errors.cpu().numpy())
        
        return np.concatenate(all_errors)
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'contamination_rate': self.contamination_rate
            }
        }, path)
    
    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.classifier.load_state_dict(checkpoint['classifier'])