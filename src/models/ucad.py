"""
UCAD: Unsupervised Cyberattack Detection
Complete Paper-Faithful PyTorch Implementation with Full Transformer Architecture

Based on: "UCAD: Unsupervised Cyberattack Detection Framework"
Architecture includes:
- Multi-layer DAE (Encoder + Decoder)
- Dynamic Pseudo-Label Generator
- Full Transformer Encoder with Multi-Head Self-Attention
- Feed-Forward Networks with GeLU activation
- End-to-End Joint Training Framework
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
import time
import math


# ============================================================
# 1. DAE COMPONENTS (Paper Section 4.1.2 - Stage 1)
# ============================================================

class DAEEncoder(nn.Module):
    """
    Denoising Autoencoder Encoder.
    Maps high-dimensional input X to low-dimensional latent space Z.
    
    Architecture: X → [Linear → ReLU → Dropout]* → Z
    Paper Equation (2): Z = Encoder(X)
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        latent_dim: int, 
        dropout: float = 0.1
    ):
        super(DAEEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final projection to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through encoder.
        Args:
            x: [batch_size, input_dim]
        Returns:
            z: [batch_size, latent_dim]
        """
        return self.encoder(x)


class DAEDecoder(nn.Module):
    """
    Denoising Autoencoder Decoder.
    Reconstructs input from latent space representation.
    
    Architecture: Z → [Linear → ReLU → Dropout]* → X_recon
    Paper Equation (3): X_recon = Decoder(Z)
    """
    def __init__(
        self, 
        latent_dim: int, 
        hidden_dims: List[int], 
        output_dim: int, 
        dropout: float = 0.1
    ):
        super(DAEDecoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Build decoder layers (mirror of encoder)
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final reconstruction layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z):
        """
        Forward pass through decoder.
        Args:
            z: [batch_size, latent_dim]
        Returns:
            x_recon: [batch_size, output_dim]
        """
        return self.decoder(z)


# ============================================================
# 2. TRANSFORMER COMPONENTS (Paper Section 4.1.2 - Stage 3)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism.
    
    Paper Equations (12-14):
    - Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
    - MultiHead(Q,K,V) = Cat(head_1, ..., head_h) * W^O
    - head_i = Attention(Q*W^Q_i, K*W^K_i, V*W^V_i)
    
    This allows the model to capture feature correlations across different subspaces.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1
    ):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V (Paper: W^Q, W^K, W^V)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection (Paper: W^O)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for scaled dot-product attention
        self.scale = math.sqrt(self.d_k)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization."""
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, mask=None):
        """
        Multi-head self-attention forward pass.
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections and reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention (Paper Equation 12)
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [batch_size, num_heads, seq_len, d_k]
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads (Paper Equation 13)
        # [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    
    Paper Equation (15): FFN(x) = Linear(GeLU(Linear(x)))
    
    Applies non-linear transformation at each position independently.
    Uses GeLU activation as specified in the paper.
    """
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super(PositionWiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Paper specifies GeLU
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Structure (as per paper):
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. Position-wise FFN + Residual + LayerNorm
    
    This is the core building block of the pseudo-classifier.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # Position-wise feed-forward
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Sub-layer 1: Multi-head self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Sub-layer 2: Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x


class PseudoClassifier(nn.Module):
    """
    Full Transformer-based Pseudo-Classifier (Paper Section 4.1.2 - Stage 3).
    
    Architecture:
    1. Takes latent space representations Z as input
    2. Passes through multiple Transformer encoder layers
    3. Extracts features using multi-head self-attention and FFN
    4. Projects to class probabilities via linear layer + LogSoftmax
    
    Paper Equations (16-17):
    - Ŷ_p = Classifier(Z_p)
    - L_NLL = -Σ Y_p^i * log(Ŷ_p^i)
    """
    def __init__(
        self, 
        latent_dim: int, 
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1
    ):
        super(PseudoClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Stack of Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(latent_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_classes)
        )
        
        # LogSoftmax for NLL loss (Paper Equation 17)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier head weights."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z):
        """
        Forward pass through pseudo-classifier.
        
        Args:
            z: Latent representations [batch_size, latent_dim]
        
        Returns:
            log_probs: Log probabilities [batch_size, num_classes]
        """
        # Expand to sequence format for Transformer
        # [batch_size, latent_dim] -> [batch_size, 1, latent_dim]
        z_expanded = z.unsqueeze(1)
        
        # Pass through Transformer layers
        for layer in self.transformer_layers:
            z_expanded = layer(z_expanded)
        
        # Extract feature representation
        # [batch_size, 1, latent_dim] -> [batch_size, latent_dim]
        features = z_expanded.squeeze(1)
        
        # Classification
        logits = self.classifier(features)
        log_probs = self.log_softmax(logits)
        
        return log_probs


# ============================================================
# 3. PSEUDO-LABEL GENERATOR (Paper Section 4.1.2 - Stage 2)
# ============================================================

class PseudoLabelGenerator:
    """
    Dynamic Pseudo-Label Generator.
    
    Paper Equations (5-9):
    - Calculates reconstruction errors for all samples
    - Sorts by error (ascending)
    - Splits based on contamination rate c
    - Assigns pseudo-labels: bottom (1-c) = normal, top c = attack
    
    Key Innovation:
    - Dynamic: Re-generates labels each epoch
    - Adaptive: Captures data distribution changes
    - Self-correcting: Reduces error accumulation
    """
    def __init__(self, contamination_rate: float = 0.076):
        """
        Args:
            contamination_rate (c): Proportion of attack samples in training data
                                   Paper notation: c in Equations (7-9)
        """
        self.contamination_rate = contamination_rate
    
    def generate_labels(
        self, 
        X: torch.Tensor, 
        X_recon: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate pseudo-labels based on reconstruction error.
        
        Implements Paper Equations (5-9):
        - (5): r_i = (1/m) * Σ(X_i,j - X_recon_i,j)²
        - (6): idx = argsort([r_1, r_2, ..., r_n])
        - (7): Z_p,nor = Z[idx[0 : n×(1-c)]]
        - (8): Z_p,att = Z[idx[n×(1-c) : n]]
        - (9): Pseudo-labels assigned
        
        Args:
            X: Original input [n_samples, n_features]
            X_recon: Reconstructed input [n_samples, n_features]
        
        Returns:
            Y_pseudo: Pseudo-labels [n_samples] (0=normal, 1=attack)
            reconstruction_errors: Array of reconstruction errors [n_samples]
        """
        # Calculate reconstruction error per sample (Paper Eq. 5)
        # Mean squared error across features
        reconstruction_errors = torch.mean((X - X_recon) ** 2, dim=1).detach().cpu().numpy()
        
        # Sort by reconstruction error ascending (Paper Eq. 6)
        sorted_indices = np.argsort(reconstruction_errors)
        
        # Calculate split point based on contamination rate (Paper Eq. 7-8)
        n_samples = len(reconstruction_errors)
        split_point = int(n_samples * (1 - self.contamination_rate))
        
        # Generate pseudo-labels (Paper Eq. 9)
        pseudo_labels = np.zeros(n_samples, dtype=np.int64)
        
        # Bottom (1-c) quantile: pseudo-normal (label 0)
        # Top c quantile: pseudo-attack (label 1)
        pseudo_labels[sorted_indices[split_point:]] = 1
        
        return torch.from_numpy(pseudo_labels), reconstruction_errors


# ============================================================
# 4. MAIN UCAD DETECTOR (Paper Section 4.1)
# ============================================================

class UCADDetector:
    """
    Complete UCAD Detector with Joint Training Framework.
    
    Paper Architecture (Figure 3):
    - DAE Encoder: Input → Latent Space
    - DAE Decoder: Latent Space → Reconstruction
    - Pseudo-Label Generator: Reconstruction Error → Labels
    - Pseudo-Classifier: Latent Space → Detection
    
    Training (Algorithm 2):
    - Stage 1: Reconstruction (train DAE)
    - Stage 2: Pseudo-label generation (dynamic labeling)
    - Stage 3: Pseudo-classification (train classifier)
    
    Detection (Algorithm 1):
    - Input → Encoder → Latent Z
    - Z → Classifier → Log Probs
    - argmax → Detection Result
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
        """
        Initialize UCAD components.
        
        Args:
            input_dim: Number of input features (m in paper)
            latent_dim: Latent space dimension (embedding_dim in paper)
            hidden_dims_encoder: Hidden layer sizes for encoder
            hidden_dims_decoder: Hidden layer sizes for decoder
            transformer_layers: Number of Transformer encoder layers
            transformer_heads: Number of attention heads
            transformer_dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            contamination_rate: Expected proportion of attacks (c in paper)
            device: 'cpu' or 'cuda'
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.contamination_rate = contamination_rate
        self.device = torch.device(device)
        
        print(f"\nInitializing UCAD with Full Transformer Architecture:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Encoder layers: {hidden_dims_encoder}")
        print(f"  Decoder layers: {hidden_dims_decoder}")
        print(f"  Transformer layers: {transformer_layers}")
        print(f"  Attention heads: {transformer_heads}")
        print(f"  FFN dimension: {transformer_dim_feedforward}")
        print(f"  Contamination rate: {contamination_rate:.1%}")
        print(f"  Device: {self.device}\n")
        
        # Initialize DAE Encoder (Paper Equation 2)
        self.encoder = DAEEncoder(
            input_dim, hidden_dims_encoder, latent_dim, dropout
        ).to(self.device)
        
        # Initialize DAE Decoder (Paper Equation 3)
        self.decoder = DAEDecoder(
            latent_dim, hidden_dims_decoder, input_dim, dropout
        ).to(self.device)
        
        # Initialize Pseudo-Classifier (Paper Equations 16-17)
        self.classifier = PseudoClassifier(
            latent_dim, 
            num_classes=2,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            d_ff=transformer_dim_feedforward,
            dropout=dropout
        ).to(self.device)
        
        # Initialize Pseudo-Label Generator (Paper Equations 5-9)
        self.pseudo_label_generator = PseudoLabelGenerator(contamination_rate)
        
        # Training history
        self.history = {
            'recon_loss': [],
            'cls_loss': [],
            'epoch_time': [],
            'pseudo_label_distribution': []
        }
        
        # Count parameters
        total_params = sum(p.numel() for p in self.encoder.parameters())
        total_params += sum(p.numel() for p in self.decoder.parameters())
        total_params += sum(p.numel() for p in self.classifier.parameters())
        print(f"Total parameters: {total_params:,}\n")
    
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
        Train UCAD using the joint end-to-end framework.
        
        Implements Algorithm 2 from the paper:
        
        for epoch in 1..N_ep:
            # Stage 1: Reconstruction
            for minibatch in training_data:
                Z = Encoder(X)
                X_recon = Decoder(Z)
                Update DAE using L_recon
            
            # Stage 2: Pseudo-label generation
            Z = Encoder(full_train_data)
            X_recon = Decoder(Z)
            Calculate reconstruction errors
            Generate pseudo-labels based on contamination rate
            
            # Stage 3: Pseudo-classification
            for minibatch in pseudo_labeled_data:
                Ŷ = Classifier(Z)
                Update Classifier using L_NLL
        
        Args:
            X_train: Training data [n_samples, n_features]
            epochs: Number of training epochs (N_ep in paper)
            batch_size: Mini-batch size (b in paper)
            lr_dae: Learning rate for DAE
            lr_classifier: Learning rate for pseudo-classifier
            patience: Early stopping patience
            verbose: Print training progress
        """
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        n_train = len(X_train)
        
        # Create data loader for reconstruction stage
        train_dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizers (Adam as specified in paper)
        optimizer_dae = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr_dae
        )
        optimizer_cls = optim.Adam(self.classifier.parameters(), lr=lr_classifier)
        
        # Loss functions
        criterion_recon = nn.MSELoss()  # Paper Equation 4
        criterion_cls = nn.NLLLoss()    # Paper Equation 17
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            print("\n" + "="*70)
            print("TRAINING UCAD - JOINT END-TO-END FRAMEWORK")
            print("Paper: Algorithm 2 - Unsupervised end-to-end joint training")
            print("="*70)
            print(f"Training samples: {n_train}")
            print(f"Contamination rate (c): {self.contamination_rate:.1%}")
            print(f"Expected normal samples: {int(n_train * (1 - self.contamination_rate))}")
            print(f"Expected attack samples: {int(n_train * self.contamination_rate)}")
            print(f"Batch size: {batch_size}")
            print(f"Epochs: {epochs}")
            print("="*70 + "\n")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # =============================================
            # STAGE 1: RECONSTRUCTION (Paper Algorithm 2, lines 3-8)
            # =============================================
            self.encoder.train()
            self.decoder.train()
            
            recon_loss_epoch = 0.0
            n_batches = 0
            
            for batch in train_loader:
                X_batch = batch[0]
                
                optimizer_dae.zero_grad()
                
                # Forward pass (Paper Equations 2-3)
                Z = self.encoder(X_batch)           # Equation 2
                X_recon = self.decoder(Z)           # Equation 3
                
                # Reconstruction loss (Paper Equation 4)
                loss_recon = criterion_recon(X_recon, X_batch)
                
                # Backward pass
                loss_recon.backward()
                optimizer_dae.step()
                
                recon_loss_epoch += loss_recon.item() * X_batch.size(0)
                n_batches += 1
            
            avg_recon_loss = recon_loss_epoch / n_train
            self.history['recon_loss'].append(avg_recon_loss)
            
            # =============================================
            # STAGE 2: PSEUDO-LABEL GENERATION (Paper Algorithm 2, lines 9-16)
            # =============================================
            self.encoder.eval()
            self.decoder.eval()
            
            with torch.no_grad():
                # Process full training set (Paper lines 10-11)
                Z_all = self.encoder(X_tensor)          # Line 10
                X_recon_all = self.decoder(Z_all)       # Line 11
            
            # Generate pseudo-labels (Paper lines 12-16, Equations 5-9)
            Y_pseudo, recon_errors = self.pseudo_label_generator.generate_labels(
                X_tensor.cpu(), X_recon_all.cpu()
            )
            
            # Track pseudo-label distribution
            pseudo_dist = np.bincount(Y_pseudo.numpy())
            self.history['pseudo_label_distribution'].append(pseudo_dist)
            
            # =============================================
            # STAGE 3: PSEUDO-CLASSIFICATION (Paper Algorithm 2, lines 17-23)
            # =============================================
            self.classifier.train()
            
            # Create data loader for pseudo-labeled data (Paper line 19-20)
            pseudo_dataset = TensorDataset(Z_all.cpu(), Y_pseudo)
            pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)
            
            cls_loss_epoch = 0.0
            
            for Z_batch, Y_batch in pseudo_loader:
                Z_batch = Z_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                optimizer_cls.zero_grad()
                
                # Forward pass (Paper Equation 16)
                log_probs = self.classifier(Z_batch)
                
                # Classification loss (Paper Equation 17)
                loss_cls = criterion_cls(log_probs, Y_batch)
                
                # Backward pass (Paper line 22)
                loss_cls.backward()
                optimizer_cls.step()
                
                cls_loss_epoch += loss_cls.item() * Z_batch.size(0)
            
            avg_cls_loss = cls_loss_epoch / n_train
            self.history['cls_loss'].append(avg_cls_loss)
            
            # =============================================
            # EPOCH SUMMARY
            # =============================================
            epoch_time = time.time() - epoch_start
            self.history['epoch_time'].append(epoch_time)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Recon Loss: {avg_recon_loss:.6f} | "
                      f"Cls Loss: {avg_cls_loss:.6f} | "
                      f"Time: {epoch_time:.1f}s")
                print(f"  Pseudo-labels → Normal: {pseudo_dist[0]}, Attack: {pseudo_dist[1]}")
                
                # Show reconstruction error statistics
                normal_errors = recon_errors[Y_pseudo.numpy() == 0]
                attack_errors = recon_errors[Y_pseudo.numpy() == 1]
                print(f"  Recon Error → Normal: {normal_errors.mean():.6f}, "
                      f"Attack: {attack_errors.mean():.6f}\n")
            
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
            print(f"Final reconstruction loss: {avg_recon_loss:.6f}")
            print(f"Final classification loss: {avg_cls_loss:.6f}")
            print("="*70 + "\n")
    
    def detect(self, X: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform detection on new data.
        
        Implements Algorithm 1 from the paper (Detection algorithm):
        
        for each sample in test_data:
            z_i = Encoder(test_sample_i)
            ŷ_i = argmax(Classifier(z_i))
        
        Paper Equation (1): 
        ŷ_i = 0 if p_0,i ≥ p_1,i (normal)
        ŷ_i = 1 if p_0,i < p_1,i (attack)
        
        Args:
            X: Input data [n_samples, n_features]
            batch_size: Batch size for inference
        
        Returns:
            predictions: Binary predictions [n_samples] (0=normal, 1=attack)
            log_probs: Log probabilities [n_samples, 2] for each class
        """
        self.encoder.eval()
        self.classifier.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        all_predictions = []
        all_log_probs = []
        
        with torch.no_grad():
            # Process in batches for memory efficiency
            for i in range(0, len(X), batch_size):
                X_batch = X_tensor[i:i+batch_size]
                
                # Encode to latent space (Paper Algorithm 1, line 2)
                Z = self.encoder(X_batch)
                
                # Classify (Paper Algorithm 1, line 3)
                log_probs = self.classifier(Z)
                
                # Get predictions (Paper Equation 1)
                predictions = torch.argmax(log_probs, dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_log_probs.append(log_probs.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        log_probs = np.concatenate(all_log_probs)
        
        return predictions, log_probs
    
    def get_latent_representations(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Get latent space representations for visualization.
        
        Extracts the learned feature representations Z from the encoder.
        Useful for t-SNE visualization and understanding learned features.
        
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
        
        Paper Equation (5): r_i = (1/m) * Σ(X_i,j - X_recon_i,j)²
        
        These errors are used to:
        1. Generate pseudo-labels during training
        2. Analyze model behavior
        3. Understand separation between normal and attack samples
        
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
                
                # Encode and decode
                Z = self.encoder(X_batch)
                X_recon = self.decoder(Z)
                
                # Calculate MSE per sample (Paper Equation 5)
                errors = torch.mean((X_batch - X_recon) ** 2, dim=1)
                all_errors.append(errors.cpu().numpy())
        
        return np.concatenate(all_errors)
    
    def get_attention_weights(self, X: np.ndarray, layer_idx: int = 0) -> np.ndarray:
        """
        Extract attention weights from Transformer layers.
        
        This method allows visualization of what the model focuses on,
        providing interpretability for the detection decisions.
        
        Args:
            X: Input data [n_samples, n_features]
            layer_idx: Which Transformer layer to extract weights from
        
        Returns:
            attention_weights: [n_samples, num_heads, seq_len, seq_len]
        """
        self.encoder.eval()
        self.classifier.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Get latent representations
            Z = self.encoder(X_tensor)
            Z_expanded = Z.unsqueeze(1)  # [batch_size, 1, latent_dim]
            
            # Forward through Transformer layers up to target layer
            if layer_idx < len(self.classifier.transformer_layers):
                layer = self.classifier.transformer_layers[layer_idx]
                
                # Get attention weights from the self-attention module
                # Note: This requires modifying the forward to return attention weights
                # For now, we'll return a placeholder
                # In practice, you'd modify MultiHeadSelfAttention to return weights
                pass
        
        # Placeholder - would need to modify forward pass to capture attention
        return None
    
    def save_model(self, path: str):
        """
        Save complete model state.
        
        Saves:
        - Encoder weights
        - Decoder weights
        - Classifier weights
        - Model configuration
        - Training history
        
        Args:
            path: Path to save model checkpoint
        """
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'contamination_rate': self.contamination_rate
            },
            'history': self.history
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model weights from checkpoint.
        
        Args:
            path: Path to model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        
        # Restore history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from {path}")
    
    def summary(self):
        """
        Print model architecture summary.
        """
        print("\n" + "="*70)
        print("UCAD MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        
        # Encoder
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"\n1. DAE Encoder:")
        print(f"   Input dim: {self.input_dim}")
        print(f"   Latent dim: {self.latent_dim}")
        print(f"   Parameters: {encoder_params:,}")
        
        # Decoder
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"\n2. DAE Decoder:")
        print(f"   Latent dim: {self.latent_dim}")
        print(f"   Output dim: {self.input_dim}")
        print(f"   Parameters: {decoder_params:,}")
        
        # Classifier
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        print(f"\n3. Pseudo-Classifier (Transformer):")
        print(f"   Input dim: {self.latent_dim}")
        print(f"   Num layers: {self.classifier.num_layers}")
        print(f"   Parameters: {classifier_params:,}")
        
        # Total
        total_params = encoder_params + decoder_params + classifier_params
        print(f"\n{'='*70}")
        print(f"Total Parameters: {total_params:,}")
        print(f"{'='*70}\n")


# ============================================================
# 5. UTILITY FUNCTIONS
# ============================================================

def test_ucad_architecture():
    """
    Test UCAD architecture with dummy data.
    Useful for debugging and verifying implementation.
    """
    print("Testing UCAD Architecture...")
    
    # Create dummy data
    n_samples = 1000
    n_features = 20
    X_dummy = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Initialize UCAD
    ucad = UCADDetector(
        input_dim=n_features,
        latent_dim=32,
        hidden_dims_encoder=[64, 128],
        hidden_dims_decoder=[128, 64],
        transformer_layers=2,
        transformer_heads=4,
        transformer_dim_feedforward=128,
        dropout=0.1,
        contamination_rate=0.1,
        device='cpu'
    )
    
    # Print summary
    ucad.summary()
    
    # Test forward pass
    print("Testing forward pass...")
    
    # Test encoder
    X_tensor = torch.FloatTensor(X_dummy[:10])
    Z = ucad.encoder(X_tensor)
    print(f"✓ Encoder output shape: {Z.shape}")
    
    # Test decoder
    X_recon = ucad.decoder(Z)
    print(f"✓ Decoder output shape: {X_recon.shape}")
    
    # Test classifier
    log_probs = ucad.classifier(Z)
    print(f"✓ Classifier output shape: {log_probs.shape}")
    
    # Test detection
    predictions, probs = ucad.detect(X_dummy[:10])
    print(f"✓ Detection predictions shape: {predictions.shape}")
    print(f"✓ Detection probabilities shape: {probs.shape}")
    
    print("\n✓ All architecture tests passed!")
    
    return ucad


if __name__ == '__main__':
    # Run architecture test
    ucad = test_ucad_architecture()