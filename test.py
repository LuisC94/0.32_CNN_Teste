#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import joblib
from loguru import logger
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import timedelta
from sqlalchemy import create_engine, text, MetaData, Table, String, Column, BigInteger, Integer, SmallInteger, Float, DateTime, Boolean
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from dotenv import load_dotenv
import warnings
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from typing import Union
import random

warnings.filterwarnings("ignore")

# Configuração do dispositivo e carregamento das variáveis de ambiente
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Usando dispositivo: {device}")
load_dotenv("env")

# =========================================================================
# === FUNÇÕES DE CÁLCULO DE MÉTRICAS (DO PLOTS1.PY) ===
# =========================================================================

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def calculate_mape(actual, prediction):
    actual = np.array(actual)
    prediction = np.array(prediction)
    non_zero_mask = actual != 0
    if not np.any(non_zero_mask):
        return 0.0 
    actual_filtered = actual[non_zero_mask]
    prediction_filtered = prediction[non_zero_mask]
    mape = np.mean(np.abs((actual_filtered - prediction_filtered) / actual_filtered)) * 100
    return mape

def calculate_custom_score(rmse, mae, bias):
    return (0.25 * bias) + (0.25 * mae) + (0.50 * rmse)

# =========================================================================
# === FUNÇÕES DE VISUALIZAÇÃO E SALVAR RESULTADOS (DO PLOTS1.PY) ===
# =========================================================================

def get_root_dir(default_value: str = ".") -> Path:
    return Path(os.getenv('ML_PIPELINE_ROOT_DIR', default_value))

class DatabaseHandler:
    def __init__(self) -> None:
        self.conn = self.start_connection()
        self.metadata = MetaData()
    
    def start_connection(self):
        return create_engine(
            f'postgresql+psycopg2://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'
        )
        
    def run_read_query(self, query_temp: str, params: dict={}):
        with self.conn.connect() as connection:
            query = text(query_temp)
            result = connection.execute(query, params)
            return pd.DataFrame(result.fetchall(), columns=result.keys())

def generate_plots_and_save_html(df_output, real_train_df, product_promos, output_filename, model_name_for_title):
    figures_to_output = []
    PROMO_ITEM_ID_COLUMN = 'item_id'
    PROMO_DATE_COLUMN = 'day'
    PROMO_TYPE_COLUMN = 'promo_type' 
    TARGET_VARIABLE = 'value'
    DATE_COLUMN = 'date'
    ITEM_ID_COLUMN = 'item_id' 
    promo_marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x-thin', 'star', 'hexagram', 'pentagon']
    promo_color_palette = ['#FF1493', '#00BFFF', '#32CD32', '#FFD700', '#BA55D3', '#FF7F50', '#40E0D0', '#DAA520']
    PROMO_MARKER_SIZE = 7

    output_directory = os.path.dirname(output_filename)
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    for item_id in df_output[ITEM_ID_COLUMN].unique():
        df_init = real_train_df[real_train_df[ITEM_ID_COLUMN] == item_id].copy()
        df_pred = df_output[df_output[ITEM_ID_COLUMN] == item_id].copy()
        df_init[DATE_COLUMN] = pd.to_datetime(df_init[DATE_COLUMN])
        df_pred[DATE_COLUMN] = pd.to_datetime(df_pred[DATE_COLUMN])
        df_init = df_init.sort_values(by=DATE_COLUMN).reset_index(drop=True)
        df_pred = df_pred.sort_values(by=DATE_COLUMN).reset_index(drop=True)
        fig = go.Figure()
        color_train_line = 'blue'
        color_actual_line = 'green'
        color_prediction_line = 'orange'
        color_prev_year_actual_line = 'purple'
        fig.add_trace(go.Scatter(x=df_init[DATE_COLUMN], y=df_init[TARGET_VARIABLE], mode='lines', name='Train', line=dict(color=color_train_line)))
        if not df_pred.empty:
            fig.add_trace(go.Scatter(x=df_pred[DATE_COLUMN], y=df_pred['actual'], mode='lines', name='Actuals', line=dict(color=color_actual_line)))
            fig.add_trace(go.Scatter(x=df_pred[DATE_COLUMN], y=df_pred['prediction'], mode='lines', name='Predictions', line=dict(color=color_prediction_line)))
            start_date_pred = df_pred[DATE_COLUMN].min()
            end_date_pred = df_pred[DATE_COLUMN].max()
            start_date_prev_year_orig = start_date_pred - pd.DateOffset(years=1)
            end_date_prev_year_orig = end_date_pred - pd.DateOffset(years=1)
            df_prev_year_segment = real_train_df[
                (real_train_df[ITEM_ID_COLUMN] == item_id) &
                (real_train_df[DATE_COLUMN] >= start_date_prev_year_orig) &
                (real_train_df[DATE_COLUMN] <= end_date_prev_year_orig)
            ].copy()
            if not df_prev_year_segment.empty:
                df_prev_year_segment = df_prev_year_segment.sort_values(by=DATE_COLUMN).reset_index(drop=True)
                plot_dates_prev_year = df_prev_year_segment[DATE_COLUMN] + pd.DateOffset(years=1)
                fig.add_trace(go.Scatter(
                    x=plot_dates_prev_year,
                    y=df_prev_year_segment[TARGET_VARIABLE],
                    mode='lines',
                    name='Prev Year Actuals (aligned)',
                    line=dict(color=color_prev_year_actual_line, dash='dashdot')
                ))
        item_promos_df = product_promos[product_promos[PROMO_ITEM_ID_COLUMN] == item_id].copy()
        if not item_promos_df.empty:
            item_promos_df[PROMO_DATE_COLUMN] = pd.to_datetime(item_promos_df[PROMO_DATE_COLUMN])
            unique_promo_types = sorted(item_promos_df[PROMO_TYPE_COLUMN].unique())
            legend_added_for_promo_type = set()
            for i, promo_type in enumerate(unique_promo_types):
                marker_symbol_for_promo = promo_marker_symbols[i % len(promo_marker_symbols)]
                marker_color_for_promo = promo_color_palette[i % len(promo_color_palette)]
                current_promo_type_dates = item_promos_df[item_promos_df[PROMO_TYPE_COLUMN] == promo_type][PROMO_DATE_COLUMN]
                show_this_promo_in_legend_for_current_set = promo_type not in legend_added_for_promo_type
                train_promo_data = df_init[df_init[DATE_COLUMN].isin(current_promo_type_dates)]
                if not train_promo_data.empty:
                    fig.add_trace(go.Scatter(x=train_promo_data[DATE_COLUMN], y=train_promo_data[TARGET_VARIABLE], mode='markers', name=promo_type, marker=dict(symbol=marker_symbol_for_promo, color=marker_color_for_promo, size=PROMO_MARKER_SIZE, line=dict(color='black', width=1)), legendgroup=promo_type, showlegend=show_this_promo_in_legend_for_current_set))
                    if show_this_promo_in_legend_for_current_set:
                        legend_added_for_promo_type.add(promo_type)
                        show_this_promo_in_legend_for_current_set = False
                if not df_pred.empty:
                    actual_promo_data = df_pred[df_pred[DATE_COLUMN].isin(current_promo_type_dates)]
                    if not actual_promo_data.empty:
                        fig.add_trace(go.Scatter(x=actual_promo_data[DATE_COLUMN], y=actual_promo_data['actual'], mode='markers', name=promo_type, marker=dict(symbol=marker_symbol_for_promo, color=marker_color_for_promo, size=PROMO_MARKER_SIZE, line=dict(color='black', width=1)), legendgroup=promo_type, showlegend=show_this_promo_in_legend_for_current_set))
                        if show_this_promo_in_legend_for_current_set:
                            legend_added_for_promo_type.add(promo_type)
                            show_this_promo_in_legend_for_current_set = False
        rmse, mae, bias, score, mape = 0.0, 0.0, 0.0, 0.0, 0.0
        if not df_pred.empty:
            rmse = calculate_rmse(df_pred['actual'], df_pred['prediction'])
            mae = calculate_mae(df_pred['actual'], df_pred['prediction'])
            bias = calculate_bias(df_pred['actual'], df_pred['prediction'])
            mape = calculate_mape(df_pred['actual'], df_pred['prediction'])
            score = calculate_custom_score(rmse, mae, bias)
        fig.update_layout(title=f'Iterative Forecast for Item: {item_id} (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, BIAS: {bias:.2f}, SCORE: {score:.2f})', xaxis_title='Date', yaxis_title=TARGET_VARIABLE, legend_title_text='Legend')
        figures_to_output.append(fig)

    if output_filename and figures_to_output:
        logger.info(f"Saving {len(figures_to_output)} graph(s) to {output_filename}...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8' />")
            f.write(f"<title>{model_name_for_title}</title>")
            f.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/latest/plotly.min.js"></script>')
            f.write("</head><body>\n")
            f.write(f"<h1 style='text-align:center;'>{model_name_for_title}</h1>")
            for i, figure_to_write in enumerate(figures_to_output):
                plot_html = pio.to_html(figure_to_write, full_html=False, include_plotlyjs=True)
                f.write(f"<div style='width:95%; margin:auto; padding-bottom:40px;'>{plot_html}</div>")
                if i < len(figures_to_output) - 1:
                    f.write("\n<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid #ddd;'>\n")
            f.write("\n</body></html>")
        logger.info(f"All graphs successfully saved to {output_filename}")
    else:
        logger.warning("No figures were generated to save.")

def save_results_to_excel_per_seed(df_test_predictions_for_current_seed, current_seed, store_id, model_base_name, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    file_name = f"{model_base_name}_Store_{store_id}_Seed_{current_seed}.xlsx"
    file_path = os.path.join(output_directory, file_name)
    
    try:
        df_test_predictions_for_current_seed.to_excel(file_path, index=False)
        logger.info(f"Resultados para a seed {current_seed} salvos em: {file_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar o ficheiro Excel para a seed {current_seed}: {e}")

# =========================================================================
# === CLASSES E FUNÇÕES DO MODELO ===
# =========================================================================

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

def create_sequences(matrix, seq_len, val_idx):
    X, y = [], []
    matrix_reordered = matrix.transpose(1, 0, 2)
    for i in range(len(matrix_reordered) - seq_len):
        sequence_data = matrix_reordered[i : i + seq_len, :, :]
        X.append(sequence_data.transpose(1, 2, 0))
        y.append(matrix_reordered[i + seq_len, val_idx, :])
    return np.array(X), np.array(y)

def test_multitask_model(model, initial_sequence, y_test_true, value_scaler, product_ids, df_full, test_start_date, features, scalers, edge_index=None):
    model.eval()
    num_forecast_steps = len(y_test_true)
    num_products = len(product_ids)
    num_channels = initial_sequence.shape[1] 
    sequence_length = initial_sequence.shape[3]
    feature_to_channel_idx = {name: i for i, name in enumerate(features)}
    value_feature_idx = feature_to_channel_idx['value']
    df_full['day'] = pd.to_datetime(df_full['day'])
    base_exog_features = [f for f in features if f != 'value' and not f.startswith('value_lag_')]
    future_base_exog_features_scaled = np.zeros((num_forecast_steps, len(base_exog_features), num_products))
    full_test_dates = pd.date_range(start=test_start_date, periods=num_forecast_steps, freq='D')
    for i, future_date in enumerate(full_test_dates):
        day_data = df_full[df_full['day'] == future_date]
        if day_data.empty:
            logger.warning(f"Aviso: Não há dados para a data {future_date} em df_full. Preenchendo features exógenas com zeros.")
            for j, _ in enumerate(base_exog_features):
                future_base_exog_features_scaled[i, j, :] = np.zeros(num_products) 
            continue
        for j, feature_name in enumerate(base_exog_features):
            current_feature_values_for_all_products = day_data.set_index('item_id').loc[product_ids, feature_name].values
            current_feature_values_for_all_products = current_feature_values_for_all_products.reshape(1, -1)
            scaled_feature_values = scalers[feature_name].transform(current_feature_values_for_all_products).flatten()
            future_base_exog_features_scaled[i, j, :] = scaled_feature_values
    all_predictions_scaled = []
    current_sequence_np = np.copy(initial_sequence)
    with torch.no_grad():
        for i in range(num_forecast_steps):
            current_sequence_tensor = torch.FloatTensor(current_sequence_np).to(device)
            input_for_model = current_sequence_tensor
            prediction_scaled = model(input_for_model, edge_index)
            all_predictions_scaled.append(prediction_scaled.cpu().numpy().flatten())
            new_day_slice = np.zeros((1, num_channels, num_products, 1))
            new_day_slice[:, value_feature_idx, :, 0] = prediction_scaled.cpu().numpy().flatten()
            current_base_exog_values_scaled_for_day_i = future_base_exog_features_scaled[i, :, :]
            for j, feature_name in enumerate(base_exog_features):
                channel_idx = feature_to_channel_idx[feature_name]
                new_day_slice[:, channel_idx, :, 0] = current_base_exog_values_scaled_for_day_i[j, :]
            for feature_name in features:
                if feature_name.startswith('value_lag_'):
                    lag_num = int(feature_name.split('_')[-1])
                    lag_channel_idx = feature_to_channel_idx[feature_name]
                    if sequence_length >= lag_num:
                        lag_value = current_sequence_np[:, value_feature_idx, :, -lag_num]
                    else:
                        lag_value = np.zeros((1, num_products))
                    new_day_slice[:, lag_channel_idx, :, 0] = lag_value.flatten()
            current_sequence_np = np.concatenate([current_sequence_np[:, :, :, 1:], new_day_slice], axis=3)
    all_predictions_scaled = np.array(all_predictions_scaled)
    predictions_np_all = value_scaler.inverse_transform(all_predictions_scaled) 
    targets_np_all = value_scaler.inverse_transform(y_test_true)
    
    df_output_list = []
    for idx, item_id in enumerate(product_ids):
        for day_idx, test_date in enumerate(full_test_dates):
            df_output_list.append({
                'item_id': item_id,
                'date': test_date,
                'actual': targets_np_all[day_idx, idx],
                'prediction': predictions_np_all[day_idx, idx]
            })
    df_output = pd.DataFrame(df_output_list)
    return df_output

# =========================================================================
# === FUNÇÃO PRINCIPAL PARA ORQUESTRAR O TESTE E AVALIAÇÃO ===
# =========================================================================
def main():
    logger.info('Iniciando o pipeline de teste e avaliação...')
    
    # === 1. CARREGAR OS DADOS E OBJETOS NECESSÁRIOS ===
    processed_data_dir = "processed_data"
    model_outputs_dir = "model_outputs"
    output_results_directory = 'resultados_previsao'
    model_name_for_excel = "CNN With Graph And Attention Mechanisms"
    store_id = 6269
    
    seeds = [12, 123, 1000, 3232, 32339, 127537, 1000000, 9827389, 71234567, 826354999]
    all_results = {}

    try:
        # Carregar dados processados (uma única vez, fora do loop)
        df_val = pd.read_parquet(os.path.join(processed_data_dir, "val_data.parquet"))
        df_test = pd.read_parquet(os.path.join(processed_data_dir, "test_data.parquet"))
        df_full = pd.read_parquet(os.path.join(processed_data_dir, "df_full_original.parquet"))
        graph_data = torch.load(os.path.join(processed_data_dir, "graph_data.pt"), weights_only=False)

        # Carregar scalers e product_ids (também uma única vez)
        scalers = joblib.load(os.path.join(model_outputs_dir, "scalers.joblib"))
        product_ids = joblib.load(os.path.join(model_outputs_dir, "product_ids.joblib"))
        
        # Obter dados de promoções (uma única vez)
        logger.info("Obtendo dados de promoções...")
        db_handler = DatabaseHandler()
        SQL_promos = f"""SELECT day, item_id, promo_value, promo_type FROM anonymous.item_promos WHERE store_id = {store_id}"""
        product_promos = db_handler.run_read_query(query_temp=SQL_promos)
        db_handler.conn.dispose()
        logger.info(f"✓ product_promos obtido com shape: {product_promos.shape}")

    except FileNotFoundError as e:
        logger.error(f"Erro: Ficheiro não encontrado - {e.filename}. Verifique se os scripts anteriores foram executados e salvaram os ficheiros corretamente.")
        return
    except Exception as e:
        logger.error(f"Erro ao obter dados de promoções: {e}")
        product_promos = pd.DataFrame(columns=['day', 'item_id', 'promo_value', 'promo_type'])
        logger.warning("Usando dados de promoções vazios.")

    # === 2. INICIALIZAR PARÂMETROS E PREPARAR DADOS ===
    sequence_length = 7
    model_features = [
        'value', 'is_weekend', 'day_of_week', 'month', 'day_of_month', 'is_close_depois_amanha',
        'is_close_Amanha', 'is_close_25dez', 'is_holiday', 'dist_to_holiday', 'log_dist_to_holiday',
        'dist_to_christmas', 'log_dist_to_christmas', 'dist_to_school_break',
        'log_dist_to_school_break', 'value_lag_1', 'value_lag_2', 'value_lag_3',
        'value_lag_4', 'value_lag_5', 'value_lag_6', 'value_lag_7', 'Natal_Log_Dist'
    ]
    num_channels = len(model_features)
    num_products = len(product_ids)
    
    value_feature_idx = model_features.index('value')
    val_matrices_list = []
    for feature_name in model_features:
        val_feature_matrix = df_val.pivot(index='day', columns='item_id', values=feature_name).loc[:, product_ids].values
        val_feature_matrix_scaled = scalers[feature_name].transform(val_feature_matrix)
        val_matrices_list.append(val_feature_matrix_scaled)
    val_matrix_scaled = np.stack(val_matrices_list, axis=0)
    X_val, _ = create_sequences(val_matrix_scaled, sequence_length, value_feature_idx)
    initial_sequence = X_val[-1:, :, :, :]
    y_test_true_matrix = df_test.pivot(index='day', columns='item_id', values='value').loc[:, product_ids].values
    y_test_for_recursive_comparison_scaled = scalers['value'].transform(y_test_true_matrix)
    test_start_date = df_test['day'].min()
    
    # Preparar df_full para os gráficos
    real_train_df = df_full[df_full['day'] < test_start_date].copy()
    real_train_df = real_train_df[['item_id', 'day', 'value']].rename(columns={'day': 'date'})

    # === 3. LOOP PARA TESTAR CADA MODELO POR SEED ===
    for seed in seeds:
        logger.info(f"=== INICIANDO TESTE PARA A SEED {seed} ===")
        
        model = MultiProductCNN_with_Attention_and_GAT(input_features=num_channels, sequence_length=sequence_length, num_products=num_products, num_channels=num_channels)
        model_filename = f"model_outputs/1_Graph_CNN_seed_{seed}.pth"
        
        try:
            model_state_dict = torch.load(model_filename)
            model.load_state_dict(model_state_dict)
            model = model.to(device)
            logger.info(f"Modelo para a seed {seed} carregado com sucesso.")
        except FileNotFoundError:
            logger.error(f"Erro: Ficheiro do modelo não encontrado para a seed {seed}: {model_filename}")
            continue

        df_test_predictions = test_multitask_model(
            model=model, initial_sequence=initial_sequence, y_test_true=y_test_for_recursive_comparison_scaled,
            value_scaler=scalers['value'], product_ids=product_ids, df_full=df_full,
            test_start_date=test_start_date, features=model_features, scalers=scalers,
            edge_index=graph_data.edge_index.to(device)
        )
        
        all_results[seed] = df_test_predictions
        logger.info(f"Teste para a seed {seed} concluído. As previsões estão em 'all_results[{seed}]'.")
        
        # Gerar gráficos e salvar HTML
        graphs_filename = f"{model_name_for_excel.replace(' ', '_')}_Seed_{seed}.html"
        generate_plots_and_save_html(
            df_output=df_test_predictions, 
            real_train_df=real_train_df, 
            product_promos=product_promos, 
            output_filename=os.path.join(output_results_directory, graphs_filename),
            model_name_for_title=f"{model_name_for_excel} - Seed {seed}"
        )
        
        # Salvar resultados em Excel
        save_results_to_excel_per_seed(
            df_test_predictions_for_current_seed=df_test_predictions,
            current_seed=seed,
            store_id=store_id,
            model_base_name=model_name_for_excel,
            output_directory=output_results_directory
        )

    logger.info("Todos os testes e geração de relatórios concluídos com sucesso!")
    logger.info("\nResultados finais (uma amostra das previsões):")
    for seed, df_pred in all_results.items():
        logger.info(f"\n--- Amostra de previsões para a seed {seed} ---")
        logger.info(df_pred.head())

if __name__ == "__main__":
    main()