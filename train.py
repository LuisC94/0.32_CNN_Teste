import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import numpy as np
from torch import nn
import pandas as pd
import datetime
import holidays
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from sqlalchemy import MetaData, text
import warnings
import json
import numpy as np
from datetime import timedelta
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import joblib
from loguru import logger
import random
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Usando dispositivo: {device}")
load_dotenv("env")

class TemporalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim=2048, dropout=0.3):
        super().__init__()
        
        # Mecanismo de Atenção
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        
        # Atenção com Conexão Residual e Normalização
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # FFN com Conexão Residual e Normalização
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class MultiProductCNN_with_Attention_and_GAT(nn.Module):
    def __init__(self, input_features, sequence_length, num_products, num_channels,
                 gat_hidden_channels=64, gat_heads=4):
        super().__init__()

        self.sequence_length = sequence_length
        self.num_products = num_products
        self.gat_heads = gat_heads
        self.gat_output_channels = gat_hidden_channels * gat_heads

        # Camadas CNN
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 7), padding=(1, 3))
        self.bn3 = nn.BatchNorm2d(256)

        # Bloco de Atenção de Canal (SE Block)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 256 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256 // 16, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # NOVO: Bloco Transformer para Atenção Temporal
        # embed_dim agora é 256, correspondendo a um único produto.
        self.temporal_transformer_block = TemporalTransformerBlock(
            embed_dim=256,
            num_heads=8,
            ffn_dim=1024
        )

        # Camada GAT
        # O GAT agora recebe as features de cada produto (256 * sequence_length)
        self.gat_layer = GATConv(
            in_channels=256 * sequence_length,
            out_channels=gat_hidden_channels,
            heads=gat_heads,
            concat=True,
            dropout=0.3
        )
        
        # Camadas de Saída (Totalmente Conectadas)
        self.multi_output = nn.Sequential(
            nn.Linear(self.gat_output_channels * num_products, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_products)
        )
    
    def forward(self, x, edge_index):
        # x shape: [batch, num_channels, num_products, sequence_length]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        B, C_in_initial, P, S = x.shape

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Atenção de Canal
        channel_weights = self.se_block(x)
        x = x * channel_weights
        
        # Processamento com o Bloco Transformer
        # De [B, C_out, P, S] para [B * P, S, C_out]
        # Cada produto é tratado como uma sequência separada.
        C_out = x.shape[1]
        x_reshaped_for_transformer = x.permute(0, 2, 3, 1).reshape(B * P, S, C_out)
        
        x_processed_by_transformer = self.temporal_transformer_block(x_reshaped_for_transformer)
        
        # Reverter o reshape para [B, P, S, C_out]
        x_processed_by_transformer = x_processed_by_transformer.reshape(B, P, S, C_out)
        
        # Processamento com a GAT
        # De [B, P, S, C_out] para [B, P, S * C_out] para a GAT
        x_for_gat = x_processed_by_transformer.reshape(B, P, -1)

        gat_outputs = []
        for i in range(B):
            node_features = x_for_gat[i]
            gat_out_i = self.gat_layer(node_features, edge_index)
            gat_outputs.append(gat_out_i)

        gat_processed_x = torch.stack(gat_outputs, dim=0)
        
        # Flatten para as camadas de saída
        x = gat_processed_x.reshape(B, -1)

        output = self.multi_output(x)
        return output
    
def train_multitask_model_with_graph(model, X_train, y_train, X_val, y_val, edge_index, epochs=500, patience=5, seed=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t), 
        batch_size=batch_size, 
        shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, edge_index) 
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t, edge_index)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
            logger.info(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                
                # Salvar o modelo com um nome de ficheiro único para cada seed
                model_filename = f"model_outputs/1_Graph_CNN_seed_{seed}.pth"
                torch.save(model.state_dict(), model_filename) 
                logger.info(f"Modelo salvo para a seed {seed}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
    return model

def create_sequences(matrix, seq_len, val_idx):
    X, y = [], []
    matrix_reordered = matrix.transpose(1, 0, 2)
    for i in range(len(matrix_reordered) - seq_len):
        sequence_data = matrix_reordered[i : i + seq_len, :, :]
        X.append(sequence_data.transpose(1, 2, 0))
        y.append(matrix_reordered[i + seq_len, val_idx, :])
    return np.array(X), np.array(y)

def main():
    logger.info('Iniciando o pipeline de treino...')
    
    # === 1. CARREGAR OS DADOS PROCESSADOS (fora do loop) ===
    processed_data_dir = "processed_data"
    
    try:
        logger.info(f"Carregando dados de treino do diretório: {processed_data_dir}")
        df_train = pd.read_parquet(os.path.join(processed_data_dir, "train_data.parquet"))
        df_val = pd.read_parquet(os.path.join(processed_data_dir, "val_data.parquet"))
        graph_data = torch.load(os.path.join(processed_data_dir, "graph_data.pt"), weights_only=False)
        product_mapping = joblib.load(os.path.join(processed_data_dir, "product_mapping.joblib"))
        
    except FileNotFoundError as e:
        logger.error(f"Erro: Ficheiro não encontrado - {e.filename}. Certifique-se de que o 'data_processing.py' foi executado e salvou os ficheiros corretamente.")
        return

    # === 2. DEFINIR PARÂMETROS E FEATURES (fora do loop) ===
    product_ids = df_train['item_id'].unique()
    num_products = len(product_ids)
    sequence_length = 7
    
    model_features = [
        'value', 'is_weekend', 'day_of_week', 'month', 'day_of_month', 'is_close_depois_amanha',
        'is_close_Amanha', 'is_close_25dez', 'is_holiday', 'dist_to_holiday', 'log_dist_to_holiday',
        'dist_to_christmas', 'log_dist_to_christmas', 'dist_to_school_break',
        'log_dist_to_school_break', 'value_lag_1', 'value_lag_2', 'value_lag_3',
        'value_lag_4', 'value_lag_5', 'value_lag_6', 'value_lag_7', 'Natal_Log_Dist'
    ]
    num_channels = len(model_features)
    value_feature_idx = model_features.index('value')

    scalers = {}
    train_matrices_list = []
    val_matrices_list = []

    for feature_name in model_features:
        scaler = MinMaxScaler()
        train_feature_matrix = df_train.pivot(index='day', columns='item_id', values=feature_name).loc[:, product_ids].values
        val_feature_matrix = df_val.pivot(index='day', columns='item_id', values=feature_name).loc[:, product_ids].values
        train_feature_matrix_scaled = scaler.fit_transform(train_feature_matrix)
        val_feature_matrix_scaled = scaler.transform(val_feature_matrix)
        scalers[feature_name] = scaler
        train_matrices_list.append(train_feature_matrix_scaled)
        val_matrices_list.append(val_feature_matrix_scaled)
    
    train_matrix_scaled = np.stack(train_matrices_list, axis=0)
    val_matrix_scaled = np.stack(val_matrices_list, axis=0)

    X_train, y_train = create_sequences(train_matrix_scaled, sequence_length, value_feature_idx)
    X_val, y_val = create_sequences(val_matrix_scaled, sequence_length, value_feature_idx)
    
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Salvar os scalers e product_ids que são comuns a todas as seeds
    model_outputs_dir = "model_outputs"
    os.makedirs(model_outputs_dir, exist_ok=True)
    joblib.dump(scalers, os.path.join(model_outputs_dir, "scalers.joblib"))
    joblib.dump(product_ids, os.path.join(model_outputs_dir, "product_ids.joblib"))
    
    # === 3. LOOP DE TREINO PARA CADA SEED ===
    seeds = [12, 123, 1000, 3232, 32339, 127537, 1000000, 9827389, 71234567, 826354999]

    for seed in seeds:
        logger.info(f"=== INICIANDO TREINO PARA A SEED {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Inicializar um NOVO modelo para cada seed
        model = MultiProductCNN_with_Attention_and_GAT(
            input_features=num_channels, 
            sequence_length=sequence_length,
            num_products=num_products,
            num_channels=num_channels 
        )
        model = model.to(device)
        
        train_multitask_model_with_graph(
            model, 
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            graph_data.edge_index.to(device),
            seed=seed # Passa o valor da seed para a função de treino
        )
        logger.info(f"=== TREINO PARA A SEED {seed} CONCLUÍDO ===")
    
    logger.info("Pipeline de treino concluído com sucesso!")

if __name__ == "__main__":
    main()