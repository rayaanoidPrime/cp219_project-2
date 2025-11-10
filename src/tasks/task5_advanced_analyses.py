import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Advanced analysis libraries
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import shared preprocessing utilities (assuming they exist as in the example)
from src.preprocessing import (
    load_and_preprocess,
    load_combined_datasets  # Assuming a function to load all data
)

import warnings
warnings.filterwarnings('ignore')

# ==============================================================
# Task 5: Optional Advanced Analyses
# ==============================================================

class LSTMAutoencoder(nn.Module):
    """
    PyTorch implementation of an LSTM Autoencoder.
    The architecture consists of an encoder and a decoder.
    """
    def __init__(self, input_dim, sequence_length, embedding_dim=32):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = sequence_length
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(64, embedding_dim, batch_first=True)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(embedding_dim, 64, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(64, input_dim, batch_first=True)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Encoding
        # The output of the first LSTM is passed as input to the second
        encoder_output1, _ = self.encoder_lstm1(x)
        _, (hidden_state, _) = self.encoder_lstm2(encoder_output1)
        
        # The hidden state shape is [num_layers, batch_size, embedding_dim].
        # We take the last layer's hidden state and unsqueeze it.
        latent_vector = hidden_state[-1, :, :].unsqueeze(1)
        
        # Repeat the latent vector `seq_len` times to feed into the decoder
        decoder_input = latent_vector.repeat(1, self.seq_len, 1)
        
        # Decoding
        decoder_output1, _ = self.decoder_lstm1(decoder_input)
        reconstruction, _ = self.decoder_lstm2(decoder_output1)
        
        return reconstruction, latent_vector.squeeze(1)

class AdvancedAnalyzer:
    """
    Explore advanced analyses for anomaly detection using PyTorch for the LSTM model.
    """
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.data: pd.DataFrame = None
        self.model: LSTMAutoencoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold: float = None

        # Create output directories
        self.fig_dir = Path('outputs/figures/task5_pytorch')
        self.table_dir = Path('outputs/tables/task5_pytorch')
        self.model_dir = Path('outputs/models/task5_pytorch')
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        print(f"\n{'='*70}")
        print("RUNNING TASK 5: ADVANCED ANALYSES (PYTORCH VERSION)")
        print(f"Using device: {self.device}")
        print(f"{'='*70}\n")

    def load_and_prepare_data(self):
        """Load and preprocess data using the centralized function."""
        print("Loading and preparing data...")
        train_df, _ = load_combined_datasets(
            data_dir=self.config['data']['raw_dir'],
            train_files=self.config['data']['train_files'],
            test_files=self.config['data'].get('test_files', {}),
            mode='full', advanced=True
        )
        self.data = train_df
        if 'base_time' in self.data.columns:
            self.data.sort_values('base_time', inplace=True)
        if 'inter_arrival' not in self.data.columns:
            raise ValueError("Feature 'inter_arrival' is required.")
        print(f"Data loaded successfully: {len(self.data)} total samples.")

    def _create_sequences(self, data: pd.Series, sequence_length: int) -> np.ndarray:
        """Create overlapping sequences from a time-series."""
        sequences = [data.iloc[i:(i + sequence_length)].values for i in range(len(data) - sequence_length + 1)]
        return np.array(sequences, dtype=np.float32)

    # ------------------- 1. Time-Series Anomaly Detection (LSTM Autoencoder) -------------------

    def time_series_anomaly_detection(self, sequence_length: int = 30, epochs: int = 50, batch_size: int = 64):
        """Train a PyTorch LSTM Autoencoder to detect anomalies in message timing."""
        print("\n" + "="*70)
        print("1. TIME-SERIES ANOMALY DETECTION (PYTORCH LSTM AUTOENCODER)")
        print("="*70)

        model_path = self.model_dir / 'lstm_autoencoder.pth'
        self.model = LSTMAutoencoder(input_dim=1, sequence_length=sequence_length, embedding_dim=32).to(self.device)

        # --- Data Preparation (Moved outside the if/else block) ---
        # This data is needed for both training and threshold calculation.
        normal_data = self.data[self.data['attack'] == 0]
        time_series = normal_data['inter_arrival'].fillna(0)
        scaler = MinMaxScaler()
        time_series_scaled = scaler.fit_transform(time_series.values.reshape(-1, 1))
        
        sequences = self._create_sequences(pd.Series(time_series_scaled.flatten()), sequence_length)
        X = torch.tensor(sequences).unsqueeze(-1) # Add feature dimension

        if model_path.exists():
            print(f"Loading existing model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))
        else:
            # --- Model Training ---
            # Split data for training and validation
            X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)
            train_dataset = TensorDataset(X_train, X_train)
            val_dataset = TensorDataset(X_val, X_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            print(f"Created {len(X_train)} training and {len(X_val)} validation sequences.")

            criterion = nn.L1Loss(reduction='mean') # MAE Loss
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            
            print("\nTraining PyTorch LSTM Autoencoder...")
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 5
            history = {'loss': [], 'val_loss': []}

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                for seq_in, seq_out in train_loader:
                    seq_in = seq_in.to(self.device)
                    optimizer.zero_grad()
                    reconstruction, _ = self.model(seq_in)
                    loss = criterion(reconstruction, seq_in)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * seq_in.size(0)
                
                avg_train_loss = train_loss / len(train_loader.dataset)
                history['loss'].append(avg_train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for seq_in, seq_out in val_loader:
                        seq_in = seq_in.to(self.device)
                        reconstruction, _ = self.model(seq_in)
                        loss = criterion(reconstruction, seq_in)
                        val_loss += loss.item() * seq_in.size(0)
                
                avg_val_loss = val_loss / len(val_loader.dataset)
                history['val_loss'].append(avg_val_loss)

                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
            
            self.model.load_state_dict(torch.load(model_path)) # Load best model
            print(f"Model trained and best state saved to {model_path}")

            # Visualize training loss (only if training occurred)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(history['loss'], label='Training Loss')
            ax.plot(history['val_loss'], label='Validation Loss')
            ax.set_title('LSTM Autoencoder Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAE Loss')
            ax.legend()
            plt.tight_layout()
            fig_path_train = self.fig_dir / 'lstm_training_history.png'
            plt.savefig(fig_path_train)
            plt.close(fig)

        # --- Post-Training / Post-Loading ---
        # Determine anomaly threshold from reconstruction error on ALL normal data
        self.model.eval()
        all_train_dataset = TensorDataset(X, X)
        all_train_loader = DataLoader(all_train_dataset, batch_size=batch_size)
        
        errors = []
        with torch.no_grad():
            for seq_in, _ in all_train_loader:
                seq_in = seq_in.to(self.device)
                reconstruction, _ = self.model(seq_in)
                error = torch.mean(torch.abs(reconstruction - seq_in), dim=[1, 2])
                errors.append(error.cpu().numpy())
        
        train_mae_loss = np.concatenate(errors)
        self.threshold = np.max(train_mae_loss)
        print(f"Anomaly detection threshold set to: {self.threshold:.4f}")
        
        # Visualize reconstruction error distribution
        plt.figure(figsize=(7, 5))
        sns.histplot(train_mae_loss, bins=50, kde=True)
        plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold = {self.threshold:.4f}')
        plt.title('Reconstruction Error on Normal Data')
        plt.xlabel('Mean Absolute Error')
        plt.legend()
        plt.tight_layout()
        fig_path_thresh = self.fig_dir / 'lstm_reconstruction_threshold.png'
        plt.savefig(fig_path_thresh)
        print(f"Saved threshold visualization to {fig_path_thresh}")
        if self.logger:
            self.logger.log_figure(plt.gcf(), "task5/lstm_reconstruction_threshold")
        plt.close()

    # ------------------- 2. Graph-Based Anomaly Analysis -------------------

    def graph_based_analysis(self):
        """Model network traffic as a graph to identify anomalous communication patterns."""
        print("\n" + "="*70)
        print("2. GRAPH-BASED ANOMALY ANALYSIS")
        print("="*70)

        if 'Source' not in self.data.columns or 'Destination' not in self.data.columns:
            print("Skipping graph analysis: Source/destination MAC addresses not found.")
            return

        # Separate normal and attack dataframes
        df_normal = self.data[self.data['attack'] == 0]
        df_attack = self.data[self.data['attack'] == 1]

        # Create graphs
        G_normal = nx.from_pandas_edgelist(df_normal, 'Source', 'Destination', create_using=nx.DiGraph())
        G_attack = nx.from_pandas_edgelist(df_attack, 'Source', 'Destination', create_using=nx.DiGraph())

        # Calculate degree centrality (a measure of node importance)
        centrality_normal = nx.degree_centrality(G_normal)
        centrality_attack = nx.degree_centrality(G_attack)

        # Convert to DataFrame for comparison
        df_centrality = pd.DataFrame([centrality_normal, centrality_attack]).T
        df_centrality.columns = ['Normal Centrality', 'Attack Centrality']
        df_centrality.fillna(0, inplace=True)
        df_centrality['Centrality Change'] = df_centrality['Attack Centrality'] - df_centrality['Normal Centrality']
        df_centrality.sort_values('Centrality Change', ascending=False, inplace=True)

        print("Top 5 nodes with the largest increase in centrality during an attack:")
        print(df_centrality.head().to_string())
        
        out_path = self.table_dir / 'graph_centrality_analysis.csv'
        df_centrality.to_csv(out_path)
        print(f"Saved centrality analysis to: {out_path}")
        if self.logger:
            self.logger.log_dataframe(df_centrality, "task5/graph_centrality_analysis")

        # Visualize the graphs
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Network Communication Graph Analysis', fontsize=16, fontweight='bold')
        
        pos_normal = nx.spring_layout(G_normal, seed=42)
        nx.draw(G_normal, pos_normal, ax=axes[0], with_labels=True, node_size=500, node_color='skyblue', font_size=8, arrows=True)
        axes[0].set_title('Normal Communication Graph', fontweight='bold')
        
        pos_attack = nx.spring_layout(G_attack, seed=42)
        nx.draw(G_attack, pos_attack, ax=axes[1], with_labels=True, node_size=500, node_color='salmon', font_size=8, arrows=True)
        axes[1].set_title('Attack Communication Graph', fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_path = self.fig_dir / 'communication_graphs.png'
        plt.savefig(fig_path)
        print(f"Saved graph visualizations to {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, "task5/communication_graphs")
        plt.close(fig)

    # ------------------- 3. Visualization of Latent Space -------------------

    def visualize_latent_space(self, sequence_length: int = 30):
        """Visualize the LSTM Autoencoder's latent space using t-SNE."""
        print("\n" + "="*70)
        print("3. VISUALIZATION OF LATENT-SPACE EMBEDDINGS (t-SNE)")
        print("="*70)

        if self.model is None:
            print("Skipping latent space visualization: LSTM model not trained.")
            return

        # --- Data Preparation for all sequences ---
        full_series = self.data['inter_arrival'].fillna(0)
        scaler = MinMaxScaler()
        full_series_scaled = scaler.fit_transform(full_series.values.reshape(-1, 1))
        
        all_sequences_np = self._create_sequences(pd.Series(full_series_scaled.flatten()), sequence_length)
        all_sequences_tensor = torch.tensor(all_sequences_np).unsqueeze(-1).to(self.device)
        
        labels = self.data['attack'].rolling(window=sequence_length).max().dropna().values
        labels = labels[len(labels) - len(all_sequences_np):] # Align labels

        # --- Latent Vector Extraction ---
        self.model.eval()
        latent_vectors_list = []
        with torch.no_grad():
            # Process in batches to avoid memory issues
            all_seq_loader = DataLoader(TensorDataset(all_sequences_tensor), batch_size=256)
            for batch in all_seq_loader:
                seq_in = batch[0]
                _, latent_vector = self.model(seq_in)
                latent_vectors_list.append(latent_vector.cpu().numpy())
        
        latent_vectors = np.concatenate(latent_vectors_list, axis=0)
        print(f"Generated {latent_vectors.shape[0]} latent vectors of dimension {latent_vectors.shape[1]}.")

        # --- t-SNE Dimensionality Reduction ---
        print("Running t-SNE... (this may take a moment)")
        # Adjust perplexity if the dataset is small
        perplexity = min(30.0, len(latent_vectors) - 1.0)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        tsne_results = tsne.fit_transform(latent_vectors)
        
        # (Plotting code remains the same)
        df_tsne = pd.DataFrame({
            'tsne-2d-one': tsne_results[:,0],
            'tsne-2d-two': tsne_results[:,1],
            'label': ['Attack' if l == 1 else 'Normal' for l in labels]
        })

        # Plot the t-SNE results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label",
            palette={"Normal": "skyblue", "Attack": "salmon"},
            data=df_tsne,
            legend="full",
            alpha=0.6
        )
        plt.title('t-SNE Visualization of LSTM Latent Space', fontweight='bold')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        fig_path = self.fig_dir / 'tsne_latent_space.png'
        plt.savefig(fig_path)
        print(f"Saved t-SNE visualization to {fig_path}")
        if self.logger:
            self.logger.log_figure(plt.gcf(), "task5/tsne_latent_space")
        plt.close()
        
        return df_tsne


# ---------------------------- Runner ----------------------------

def run_task5(config: Dict[str, Any], logger=None) -> Dict:
    """
    Execute Task 5: Advanced Analyses.
    """
    results = {}
    
    analyzer = AdvancedAnalyzer(config, logger)
    
    # Load and prepare data
    analyzer.load_and_prepare_data()
    
    # 1. Run time-series anomaly detection
    analyzer.time_series_anomaly_detection()
    
    # 2. Run graph-based analysis
    analyzer.graph_based_analysis()
    
    # 3. Visualize latent space
    tsne_df = analyzer.visualize_latent_space()
    
    results['status'] = 'completed'
    results['tsne_results'] = tsne_df
    
    print(f"\n✓ Task 5 (Advanced Analyses) completed successfully!")
    print(f"All outputs saved to {analyzer.fig_dir}/ and {analyzer.table_dir}/")
    
    return results


if __name__ == '__main__':
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    results = run_task5(config, logger=None)