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
import random
import networkx as nx
import plotly.graph_objects as go
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import joblib
from loguru import logger

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Usando dispositivo: {device}")

def fill_missing_days(df, start_date='2021-01-23', end_date='2023-02-22'):
    """
    Preenche dias em falta entre start_date e end_date para cada produto
    """
    df = df.copy()
    df['day'] = pd.to_datetime(df['day'])
    
    # Range completo de datas
    complete_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df_unified = []
    
    for item_id in df['item_id'].unique():
        produto_data = df[df['item_id'] == item_id].copy()
        produto_data = produto_data.drop_duplicates(['day', 'item_id'], keep='first')
        
        # Pegar store_id do produto
        store_id = produto_data['store_id'].iloc[0]
        
        # DataFrame completo para este produto
        product_complete = pd.DataFrame({
            'day': complete_date_range,
            'item_id': item_id,
            'store_id': store_id
        })
        
        # Merge com dados existentes
        product_complete = product_complete.merge(
            produto_data[['day', 'item_id', 'value', 'promo_value', 'promo_type']], 
            on=['day', 'item_id'], 
            how='left'
        )
        
        # Preencher valores em falta
        product_complete['value'] = product_complete['value'].fillna(0)
        product_complete['promo_value'] = product_complete['promo_value'].fillna(0)
        product_complete['promo_type'] = product_complete['promo_type'].fillna('')
        
        df_unified.append(product_complete)
    
    return pd.concat(df_unified, ignore_index=True)

####################data_complete = fill_missing_days(data)

def find_nearest_christmas_date(day_series):
    """
    Função vetorizada para encontrar a data do Natal mais próxima.
    """
    natal_this_year = pd.to_datetime(day_series.dt.year.astype(str) + '-12-25')
    natal_next_year = pd.to_datetime((day_series.dt.year + 1).astype(str) + '-12-25')
    
    # Calcular a distância em dias para o Natal deste ano e do próximo
    dist_this = (day_series - natal_this_year).dt.days.values
    dist_next = (day_series - natal_next_year).dt.days.values
    
    # Criar uma máscara para decidir qual data de Natal é a mais próxima
    is_next_year_closer = np.abs(dist_next) < np.abs(dist_this)
    
    # Usar a máscara para selecionar a data de Natal correta de forma vetorizada
    nearest_christmas_dates = natal_this_year.copy()
    nearest_christmas_dates[is_next_year_closer] = natal_next_year[is_next_year_closer]
    
    return nearest_christmas_dates

def create_product_graph_no_store_2(
    df_for_attributes: pd.DataFrame, 
    item_id_column: str, 
    value_column: str,
    device='cpu'
):
    """
    Creates a product graph where nodes are unique products (item) and edges
    are formed based on high Spearman correlation of 'value'.

    Args:
        df_for_attributes (pd.DataFrame): DataFrame containing product data, including
            item IDs, dmn_ids, and value.
        item_id_column (str): Name of the column containing item IDs.
        value_column (str): Name of the column containing the sales values.
        device (str, optional): Device to store the graph data (e.g., 'cpu', 'cuda').
            Defaults to 'cpu'.

    Returns:
        Data: A PyTorch Geometric Data object representing the product graph.
        dict: A dictionary mapping product IDs to node indices. Returns an empty dict
              if no graph is created.
    """
    logger.info("--- Creating Product Graph (Spearman Correlation) ---")
    logger.info("1. Preparing Product Data and Identifiers")

    if df_for_attributes.empty:
        logger.info("Warning: Input DataFrame 'df_for_attributes' is empty. Returning empty graph.")
        return Data(x=torch.empty((0, 0), dtype=torch.float32).to(device),
                    edge_index=torch.empty((2, 0), dtype=torch.long).to(device),
                    edge_attr=torch.empty((0, 1), dtype=torch.float32).to(device)), {}

    required_columns = [item_id_column, value_column]
    missing_columns = [col for col in required_columns if col not in df_for_attributes.columns]
    if missing_columns:
        raise ValueError(f"DataFrame 'df_for_attributes' must contain the following columns: {', '.join(missing_columns)}.")

    df_processed = df_for_attributes.copy()
    df_processed['unique_product_id'] = df_processed[item_id_column].astype(str)

    product_values_series = df_processed.groupby('unique_product_id')[value_column].apply(list)
    
    # Filter out products that don't have enough values for correlation
    product_values = {
        prod_id: values
        for prod_id, values in product_values_series.items()
        if values is not None and len(values) > 1
    }

    all_product_ids = list(product_values.keys())
    n_products = len(all_product_ids)
    logger.info(f"   Number of unique products (nodes) with sufficient data: {n_products}")

    if n_products == 0:
        logger.info("Warning: No unique products found with sufficient data for correlation. Returning empty graph.")
        return Data(x=torch.empty((0, 0), dtype=torch.float32).to(device),
                    edge_index=torch.empty((2, 0), dtype=torch.long).to(device),
                    edge_attr=torch.empty((0, 1), dtype=torch.float32).to(device)), {}

    logger.info("2. Creating Edge List Based on Spearman Correlation")
    edges = []

    max_len = max(len(v) for v in product_values.values())

    padded_product_values = {
        prod_id: values + [0] * (max_len - len(values)) 
        for prod_id, values in product_values.items()
    }
    
    df_values = pd.DataFrame(padded_product_values)
    
    if df_values.empty:
        logger.info("No product values available for correlation calculation after filtering. Returning empty graph.")
        return Data(x=torch.empty((0, 0), dtype=torch.float32).to(device),
                    edge_index=torch.empty((2, 0), dtype=torch.long).to(device),
                    edge_attr=torch.empty((0, 1), dtype=torch.float32).to(device)), {}

    corr_matrix = df_values.corr(method='spearman')
    corr_matrix_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_series = corr_matrix_upper.stack()
    high_corr_pairs = high_corr_series[high_corr_series > 0.26]

    for (prod_id_i, prod_id_j), correlation in high_corr_pairs.items():
        edges.append((prod_id_i, prod_id_j, {'weight': correlation, 'reason': 'high_correlation', 'correlation': correlation}))
        logger.info(f"   Edge: {prod_id_i} -- {prod_id_j}  Rule: high_correlation, Correlation: {correlation:.4f}") 

    logger.info(f"   Number of edges based on high Spearman correlation: {len(edges)}")

    logger.info("3. Building Full Graph using NetworkX")
    G = nx.Graph()
    G.add_nodes_from(all_product_ids)
    G.add_edges_from([(u, v, data) for u, v, data in edges])

    def visualize_graph(graph, title, node_count, edge_count):
        if 0 < node_count < 3000:
            logger.info(f"   Visualizing graph (nodes: {node_count})...")
            pos = nx.spring_layout(graph, seed=42)
            fig = go.Figure()

            node_x, node_y, node_text = [], [], []
            for node_id in graph.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node_id))

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, text=node_text,
                mode='markers+text',
                textposition="bottom center",
                marker=dict(size=10, color='skyblue', line_width=1),
                name='Products'
            ))

            edge_x, edge_y = [], []
            for edge_ends in graph.edges():
                x0, y0 = pos[edge_ends[0]]
                x1, y1 = pos[edge_ends[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                name='Connections'
            ))

            fig.update_layout(
                title_text=title,
                title_x=0.5,
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                margin=dict(b=20, l=5, r=5, t=40),
                hovermode='closest'
            )
            fig.show()
        elif node_count >= 3000:
            logger.info(f"   Graph visualization skipped due to large number of nodes ({node_count} >= 3000).")

    visualize_graph(G, 'Product Graph (Spearman Correlation)', G.number_of_nodes(), G.number_of_edges())

    logger.info(f"   Number of nodes in NetworkX graph: {G.number_of_nodes()}")

    if G.number_of_nodes() > 0:
        logger.info("   Identifying connected components...")
        connected_components = list(nx.connected_components(G))
        if connected_components:
            largest_component_nodes = max(connected_components, key=len)
            G = G.subgraph(largest_component_nodes).copy()
            logger.info(f"   Found {len(connected_components)} component(s). Filtered graph to largest component with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        else:
            logger.info("   Warning: No connected components found, though graph has nodes. This should not happen if nodes exist. Using original graph.")
    
    final_product_ids_in_graph = list(G.nodes())
    n_final_nodes = G.number_of_nodes()

    if n_final_nodes == 0:
        logger.info("Warning: Graph is empty after filtering (or was initially empty). Returning empty PyG Data object.")
        return Data(x=torch.empty((0, 0), dtype=torch.float32).to(device),
                    edge_index=torch.empty((2, 0), dtype=torch.long).to(device),
                    edge_attr=torch.empty((0, 1), dtype=torch.float32).to(device)), {}

    logger.info(f"   Final graph (largest component): {n_final_nodes} nodes, {G.number_of_edges()} edges.")

    visualize_graph(G, f'Product Graph (Largest Component: {n_final_nodes} nodes, {G.number_of_edges()} edges)', n_final_nodes, G.number_of_edges())

    logger.info("4. Converting to PyTorch Geometric Data")
    item_id_to_node_idx_map = {label: i for i, label in enumerate(final_product_ids_in_graph)}

    x_features = torch.empty((n_final_nodes, 0), dtype=torch.float32).to(device) 

    edge_list_indices_one_direction = []
    edge_attr_list_temp_one_direction = []

    for u_pid, v_pid, edge_data in G.edges(data=True):
        u_idx = item_id_to_node_idx_map[u_pid]
        v_idx = item_id_to_node_idx_map[v_pid]
        edge_list_indices_one_direction.append([u_idx, v_idx])
        edge_attr_list_temp_one_direction.append([edge_data['correlation']])

    if edge_list_indices_one_direction:
        edge_index_one_dir_tensor = torch.tensor(edge_list_indices_one_direction, dtype=torch.long).t().contiguous()
        edge_attr_one_dir_tensor = torch.tensor(edge_attr_list_temp_one_direction, dtype=torch.float32)

        if edge_attr_one_dir_tensor.dim() == 1 and edge_attr_one_dir_tensor.numel() > 0:
            edge_attr_one_dir_tensor = edge_attr_one_dir_tensor.unsqueeze(1)
    
        edge_index_tensor, edge_attr_tensor = to_undirected(
            edge_index_one_dir_tensor.to('cpu'), 
            edge_attr=edge_attr_one_dir_tensor.to('cpu'),
            num_nodes=n_final_nodes
        )
        edge_index_tensor = edge_index_tensor.to(device)
        edge_attr_tensor = edge_attr_tensor.to(device)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long).to(device)
        edge_attr_tensor = torch.empty((0, 1), dtype=torch.float32).to(device)
    
    if edge_attr_tensor is None or (edge_attr_tensor.numel() == 0 and edge_index_tensor.shape[1] > 0):
        if edge_index_tensor.shape[1] > 0:
            logger.info("Warning: Edge attributes were None or empty after to_undirected, creating default attributes.")
            edge_attr_tensor = torch.ones((edge_index_tensor.shape[1], 1), dtype=torch.float32).to(device)
        else:
            edge_attr_tensor = torch.empty((0, 1), dtype=torch.float32).to(device)


    graph_data_obj = Data(x=x_features, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
    graph_data_obj.num_nodes = n_final_nodes

    try:
        nx.write_graphml(G, "my_graph.graphml")
        logger.info("Graph saved successfully as my_graph.graphml")
    except Exception as e:
        logger.info(f"Error saving graph as GraphML: {e}")

    return graph_data_obj, item_id_to_node_idx_map

def simple_traverse_by_correlation(graph_data, product_mapping):
    """
    Função simplificada que percorre o grafo seguindo as maiores correlações.
    Trabalha diretamente com edge_index e edge_attr.
    """
    import torch
    import numpy as np
    
    # Converter para numpy se necessário
    edge_index = graph_data.edge_index.numpy() if isinstance(graph_data.edge_index, torch.Tensor) else graph_data.edge_index
    edge_attr = graph_data.edge_attr.numpy() if isinstance(graph_data.edge_attr, torch.Tensor) else graph_data.edge_attr
    
    # Se edge_attr tem shape (N, 1), flatten para (N,)
    if len(edge_attr.shape) > 1:
        edge_attr = edge_attr.flatten()
    
    # Mapeamento reverso
    idx_to_product = {v: k for k, v in product_mapping.items()}
    
    logger.info(f"=== PERCURSO SIMPLIFICADO POR CORRELAÇÃO ===")
    logger.info(f"Nós: {graph_data.num_nodes}, Arestas: {len(edge_attr)}")
    logger.info(f"edge_attr shape: {edge_attr.shape}")
    logger.info(f"edge_attr range: {edge_attr.min():.4f} a {edge_attr.max():.4f}")
    logger.info(f"Primeiros 5 valores: {edge_attr[:5]}")
    
    # Criar lista simples: (correlação, nó1, nó2)
    all_edges = []
    for i in range(edge_index.shape[1]):  # Iterar sobre colunas (cada aresta)
        node1 = int(edge_index[0, i])
        node2 = int(edge_index[1, i])
        correlation = float(edge_attr[i])
        all_edges.append((correlation, node1, node2))
    
    # Ordenar por correlação (maior primeiro)
    all_edges.sort(key=lambda x: x[0], reverse=True)
    
    logger.info(f"\nTop 5 correlações:")
    for i in range(min(5, len(all_edges))):
        corr, n1, n2 = all_edges[i]
        prod1 = idx_to_product.get(n1, f"Unknown_{n1}")
        prod2 = idx_to_product.get(n2, f"Unknown_{n2}")
        logger.info(f"  {corr:.4f}: {prod1} ↔ {prod2}")
    
    # Algoritmo simples: começar com maior correlação e ir adicionando
    visited_nodes = set()
    ordered_products = []
    path_correlations = []
    
    # Adicionar primeira aresta (maior correlação)
    if all_edges:
        first_corr, first_n1, first_n2 = all_edges[0]
        visited_nodes.add(first_n1)
        visited_nodes.add(first_n2)
        ordered_products.append(idx_to_product.get(first_n1, f"Unknown_{first_n1}"))
        ordered_products.append(idx_to_product.get(first_n2, f"Unknown_{first_n2}"))
        path_correlations.append(first_corr)
        current_frontier = {first_n1, first_n2}  # Nós na "fronteira" do caminho
        
        logger.info(f"\nInício: {ordered_products[0]} ↔ {ordered_products[1]} (corr: {first_corr:.4f})")
    
    # Continuar adicionando nós
    for step in range(2, graph_data.num_nodes):
        best_edge = None
        best_correlation = -1
        
        # Encontrar melhor aresta que conecta a fronteira a um nó novo
        for corr, n1, n2 in all_edges:
            # Se um nó está na fronteira e o outro é novo
            if (n1 in current_frontier and n2 not in visited_nodes):
                if corr > best_correlation:
                    best_correlation = corr
                    best_edge = (corr, n1, n2)
            elif (n2 in current_frontier and n1 not in visited_nodes):
                if corr > best_correlation:
                    best_correlation = corr
                    best_edge = (corr, n2, n1)  # Swap para manter consistência
        
        if best_edge is None:
            logger.info(f"Sem mais conexões na fronteira no passo {step}")
            # Tentar começar novo componente
            for corr, n1, n2 in all_edges:
                if n1 not in visited_nodes and n2 not in visited_nodes:
                    best_edge = (corr, n1, n2)
                    current_frontier = {n1, n2}  # Nova fronteira
                    break
            
            if best_edge is None:
                logger.info(f"Nenhuma aresta disponível. Parando no passo {step}")
                break
        
        # Adicionar novo nó
        corr, connected_node, new_node = best_edge
        
        if new_node not in visited_nodes:
            visited_nodes.add(new_node)
            ordered_products.append(idx_to_product.get(new_node, f"Unknown_{new_node}"))
            path_correlations.append(corr)
            current_frontier.add(new_node)  # Expandir fronteira
            
            if step % 20 == 0:
                logger.info(f"Passo {step}: Adicionado {idx_to_product.get(new_node, f'Unknown_{new_node}')} (corr: {corr:.4f})")
    
    logger.info(f"\nPercurso completo: {len(ordered_products)} produtos")
    return ordered_products, path_correlations

def analyze_simple_path(ordered_products, path_correlations):
    """
    Análise simples do caminho criado
    """
    import numpy as np
    
    logger.info(f"\n=== ANÁLISE DO CAMINHO ===")
    logger.info(f"Total de produtos: {len(ordered_products)}")
    logger.info(f"Total de correlações: {len(path_correlations)}")
    
    if path_correlations:
        corr_array = np.array(path_correlations)
        logger.info(f"Correlação média: {corr_array.mean():.4f}")
        logger.info(f"Correlação min: {corr_array.min():.4f}")
        logger.info(f"Correlação max: {corr_array.max():.4f}")
        logger.info(f"Desvio padrão: {corr_array.std():.4f}")
        
        logger.info(f"\nPrimeiros 10 produtos do caminho:")
        for i in range(min(10, len(ordered_products)-1)):
            if i < len(path_correlations):
                logger.info(f"  {i+1}. {ordered_products[i]} → {ordered_products[i+1]} (corr: {path_correlations[i]:.4f})")
        
        # Distribuição simples
        if corr_array.max() > 0.7:
            logger.info(f"\nDistribuição das correlações:")
            logger.info(f"  0.70-0.71: {np.sum((corr_array >= 0.70) & (corr_array < 0.71))}")
            logger.info(f"  0.71-0.72: {np.sum((corr_array >= 0.71) & (corr_array < 0.72))}")
            logger.info(f"  0.72-0.73: {np.sum((corr_array >= 0.72) & (corr_array < 0.73))}")
            logger.info(f"  0.73-0.74: {np.sum((corr_array >= 0.73) & (corr_array < 0.74))}")
            logger.info(f"  0.74+: {np.sum(corr_array >= 0.74)}")
        else:
            logger.info(f"  Todas as correlações: {np.unique(corr_array)}")
    
    return ordered_products, path_correlations

def debug_and_reorganize(df, ordered_products_list):
    """
    Debugga diferenças e reorganiza DataFrame.
    """
    import pandas as pd
    
    logger.info("=== DEBUG ===")
    logger.info(f"Produtos na lista ordenada: {len(ordered_products_list)}")
    logger.info(f"Primeiros 5 da lista: {ordered_products_list[:5]}")
    logger.info(f"Tipo dos produtos na lista: {type(ordered_products_list[0])}")
    
    logger.info(f"\nProdutos únicos no DataFrame: {df['item_id'].nunique()}")
    produtos_df = df['item_id'].unique()
    logger.info(f"Primeiros 5 do DataFrame: {produtos_df[:5]}")
    logger.info(f"Tipo dos produtos no DataFrame: {type(produtos_df[0])}")
    
    # Verificar intersecção
    produtos_lista_str = [str(p) for p in ordered_products_list]
    produtos_df_str = [str(p) for p in produtos_df]
    
    intersecao = set(produtos_lista_str).intersection(set(produtos_df_str))
    logger.info(f"\nIntersecção (como string): {len(intersecao)} produtos")
    
    if len(intersecao) > 0:
        logger.info(f"Primeiros 5 em comum: {list(intersecao)[:5]}")
    else:
        logger.info("NENHUM produto em comum!")
        logger.info(f"Exemplo lista: {produtos_lista_str[:3]}")
        logger.info(f"Exemplo DataFrame: {produtos_df_str[:3]}")
        return None
    
    # Tentar filtrar convertendo ambos para string
    df_copy = df.copy()
    df_copy['item_id_str'] = df_copy['item_id'].astype(str)
    
    df_filtered = df_copy[df_copy['item_id_str'].isin(produtos_lista_str)].copy()
    
    if len(df_filtered) == 0:
        logger.info("Ainda não funcionou após conversão para string!")
        return None
    
    # Criar mapeamento e ordenar
    product_order = {str(product): idx for idx, product in enumerate(ordered_products_list)}
    df_filtered['order_position'] = df_filtered['item_id_str'].map(product_order)
    df_reorganized = df_filtered.sort_values(['order_position', 'day'])
    
    # Manter item_id original
    df_final = df_reorganized.drop(['item_id_str', 'order_position'], axis=1)
    
    return df_final

def get_train_test_split_dfs_silent(df, train_end_str, test_start_str):
    """
    Divide o DataFrame em conjuntos de treino e teste por datas, sem mensagens informativas.

    Args:
        df: DataFrame com coluna 'day' e já com as features diretas (não-lag) criadas.
            Deve estar pré-ordenado por 'product_id' e 'day'.
        train_end_str: String da última data de treino (inclusive).
        test_start_str: String da primeira data de teste (inclusive).

    Returns:
        df_train_raw, df_test_raw: DataFrames divididos.
    """
    train_end_date = pd.to_datetime(train_end_str)
    test_start_date = pd.to_datetime(test_start_str)

    # Dividir o DataFrame com base nas datas
    df_train = df[df['day'] <= train_end_date].copy()
    df_test = df[df['day'] >= test_start_date].copy()

    return df_train, df_test

def create_lag_features(df_input, feature_name='value', lags=[1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29, 30, 364, 365, 366]):
    """
    Cria apenas features de lag para um DataFrame.
    Deve ser aplicado após a divisão inicial do dataset.

    Args:
        df_input: DataFrame com 'day', 'product_id' (ou 'item_id'/'store_id' como no seu df)
                  e a 'feature_name' (ex: 'value').
                  Deve estar ordenado por 'product_id' (ou o ID da sua série temporal) e 'day'.
        feature_name: Nome da coluna a partir da qual os lags serão criados (ex: 'value').
        lags: Lista de lags a serem criados (ex: [1, 7, 14, 28, 364]).

    Returns:
        df_processed: DataFrame com as novas features de lag.
    """
    df_processed = df_input.copy()
    # Garante que o DataFrame está ordenado, crucial para lags
    # Use 'item_id' e 'store_id' conforme o seu DataFrame de exemplo
    #df_processed = df_processed.sort_values(by=['item_id', 'store_id', 'day'])

    # Criar lags pontuais
    for lag in lags:
        # Assumindo que você agrupa por item_id e store_id para formar a série única
        df_processed[f'{feature_name}_lag_{lag}'] = \
            df_processed.groupby(['item_id', 'store_id'])[feature_name].shift(lag)

    # Preenchimento de NaNs:
    # Primeiro, ffill (forward fill) para preencher NaNs com o último valor válido por série.
    # Isso é útil para lags que podem ter NaNs no início de cada série temporal combinada.
    for col in df_processed.columns:
        if 'lag' in col: # Apenas para colunas de lag
            df_processed[col] = df_processed.groupby(['item_id', 'store_id'])[col].ffill()

    # Segundo, preencher quaisquer NaNs remanescentes (que não puderam ser ffilled) com 0.
    # Isso pode acontecer no início absoluto da série histórica para determinados lags.
    df_processed.fillna(0, inplace=True)

    return df_processed

def create_val_from_train_silent(df_train, val_days):
    """
    Cria um DataFrame de validação a partir do final do DataFrame de treino,
    e ajusta o DataFrame de treino original para remover esses dias.
    Esta versão é simplificada e não inclui verificações ou mensagens de logger.info.

    Args:
        df_train (pd.DataFrame): O DataFrame de treino original, com a coluna 'day'.
                                 Deve estar ordenado por 'day' (e 'item_id' se aplicável).
        val_days (int): O número de dias a serem usados para o conjunto de validação,
                        retirados do final do df_train.

    Returns:
        tuple: (df_train_adjusted, df_val).
               df_train_adjusted: O DataFrame de treino sem os dias de validação.
               df_val: O novo DataFrame de validação.
    """
    # Garante que 'day' é datetime e o DataFrame está ordenado
    df_train['day'] = pd.to_datetime(df_train['day'])
    df_train_sorted = df_train.copy()

    # Encontra a última data no df_train
    last_train_date = df_train_sorted['day'].max()

    # Calcula a data de início da validação
    val_start_date = last_train_date - timedelta(days=val_days - 1)

    # Cria o df_val com base nas datas
    df_val = df_train_sorted[df_train_sorted['day'] >= val_start_date].copy()

    # Ajusta o df_train para remover os dias de validação
    df_train_adjusted = df_train_sorted[df_train_sorted['day'] < val_start_date].copy()

    return df_train_adjusted, df_val

def main():
        
    data = pd.read_csv('input_data/df.csv')

    logger.info(data.columns)

    logger.info('\n{}'.format(data))

    data_complete = fill_missing_days(data)

    non_zero_df = data_complete[data_complete['value'] != 0]
    non_zero_day_counts = non_zero_df.groupby('item_id')['day'].nunique()
    ids_to_keep = non_zero_day_counts[non_zero_day_counts >= 600].index
    data_complete = data_complete[data_complete['item_id'].isin(ids_to_keep)]
    data_complete.reset_index(drop=True, inplace=True)
    data_complete.drop(columns=['promo_value', 'promo_type'], inplace=True)

    data_complete['is_weekend'] = (data_complete['day'].dt.weekday >= 5).astype(int)
    data_complete['day_of_week'] = data_complete['day'].dt.weekday
    data_complete['month'] = data_complete['day'].dt.month
    data_complete['day_of_month'] = data_complete['day'].dt.day

    # Feriados 2021-2023
    us_holidays_2021 = holidays.UnitedStates(years=2021)
    us_holidays_2022 = holidays.UnitedStates(years=2022)
    us_holidays_2023 = holidays.UnitedStates(years=2023)

    all_holidays = []
    all_holidays.extend([pd.to_datetime(d) for d in us_holidays_2021])
    all_holidays.extend([pd.to_datetime(d) for d in us_holidays_2022])
    all_holidays.extend([pd.to_datetime(d) for d in us_holidays_2023])

    data_complete['is_holiday'] = data_complete['day'].isin(all_holidays).astype(int)

    # --- INÍCIO DAS OTIMIZAÇÕES E CORREÇÃO DO ERRO ---

    # Definir uma data de referência (época) para calcular os dias a partir dela
    # Usar uma data de referência garante que todos os cálculos de distância sejam consistentes
    epoch_date = pd.to_datetime('1970-01-01')

    # Converter as datas do DataFrame para o número de dias desde a época
    # Isso é vetorizado e eficiente
    day_numbers = (data_complete['day'] - epoch_date).dt.days.values

    # Feriados
    # Converter as datas de feriado para o número de dias desde a mesma época
    all_holidays_numbers = np.array([(d - epoch_date).days for d in all_holidays])
    # Calcular a diferença absoluta de cada dia para cada feriado e pegar o mínimo
    data_complete['dist_to_holiday'] = np.min(np.abs(day_numbers[:, None] - all_holidays_numbers), axis=1)
    data_complete['log_dist_to_holiday'] = np.log1p(data_complete['dist_to_holiday'])

    # Natal 2021-2023
    christmas_list = [pd.to_datetime('2021-12-25'), pd.to_datetime('2022-12-25'), pd.to_datetime('2023-12-25')]
    christmas_numbers = np.array([(d - epoch_date).days for d in christmas_list])
    data_complete['dist_to_christmas'] = np.min(np.abs(day_numbers[:, None] - christmas_numbers), axis=1)
    data_complete['log_dist_to_christmas'] = np.log1p(data_complete['dist_to_christmas'])

    # Férias escolares (junho-agosto)
    school_breaks = pd.date_range('2021-06-01', '2021-08-31').tolist() + \
                    pd.date_range('2022-06-01', '2022-08-31').tolist() + \
                    pd.date_range('2023-06-01', '2023-08-31').tolist()
    school_breaks_numbers = np.array([(d - epoch_date).days for d in school_breaks])
    data_complete['dist_to_school_break'] = np.min(np.abs(day_numbers[:, None] - school_breaks_numbers), axis=1)
    data_complete['log_dist_to_school_break'] = np.log1p(data_complete['dist_to_school_break'])

    # --- FIM DAS OTIMIZAÇÕES E CORREÇÃO DO ERRO ---

    # Cria as três colunas binárias para os dias 23, 24 e 25 de dezembro (já eram eficientes)
    data_complete['is_close_depois_amanha'] = (
        (data_complete['day'].dt.month == 12) &
        (data_complete['day'].dt.day == 23)
    ).astype(int)

    data_complete['is_close_Amanha'] = (
        (data_complete['day'].dt.month == 12) &
        (data_complete['day'].dt.day == 24)
    ).astype(int)

    data_complete['is_close_25dez'] = (
        (data_complete['day'].dt.month == 12) &
        (data_complete['day'].dt.day == 25)
    ).astype(int)

    # 1. Obter a data do Natal mais próximo para cada dia
    data_complete['nearest_christmas'] = find_nearest_christmas_date(data_complete['day'])

    # 2. Calcular a distância d_1 (com sinal)
    data_complete['d_1'] = (data_complete['day'] - data_complete['nearest_christmas']).dt.days

    # 3. Aplicar a lógica do afastamento logarítmico de forma vetorizada
    # Usamos a função `np.where` para aplicar a lógica condicional de forma eficiente
    data_complete['Natal_Log_Dist'] = np.where(
        data_complete['d_1'] < 0,
        np.exp(-0.1 * np.abs(data_complete['d_1'])),
        np.exp(-0.3 * np.abs(data_complete['d_1']))
    )

    # Opcional: remover as colunas auxiliares
    data_complete = data_complete.drop(columns=['nearest_christmas', 'd_1'])

    # Depois de criar o grafo (que já filtra para maior componente)
    graph_data, product_mapping = create_product_graph_no_store_2(
        df_for_attributes=data_complete,
        item_id_column='item_id', 
        value_column='value',
        device='cpu'
    )

    # Uso simples:
    logger.info("EXECUTANDO PERCURSO SIMPLIFICADO...")
    simple_products, simple_correlations = simple_traverse_by_correlation(graph_data, product_mapping)
    analyze_simple_path(simple_products, simple_correlations)

    logger.info(f"\n=== LISTA ORDENADA COMPLETA ===")
    logger.info(f"Total de produtos: {len(simple_products)}")
    logger.info("\nLista ordenada por correlação:")
    for i, product in enumerate(simple_products, 1):
        if i <= len(simple_correlations):
            corr = simple_correlations[i-1] if i > 1 else "início"
            logger.info(f"{i:2d}. {product} (corr: {corr})")
        else:
            logger.info(f"{i:2d}. {product}")

    df_reorganized = debug_and_reorganize(data_complete, simple_products)

    if df_reorganized is not None:
        logger.info(f"\n=== SUCESSO ===")
        logger.info(f"DataFrame reorganizado: {df_reorganized.shape}")
        logger.info(f"Produtos únicos: {df_reorganized['item_id'].nunique()}")
    else:
        logger.info("\n=== FALHOU - precisa investigar mais ===")

    logger.info(df_reorganized)

    data_complete_sorted = df_reorganized.copy()

    train_end_date_str = '2022-09-23'
    test_start_date_str = '2022-09-24'

    df_train, df_test = get_train_test_split_dfs_silent(
        data_complete_sorted,
        train_end_str=train_end_date_str,
        test_start_str=test_start_date_str
    )

    df_train_processed = create_lag_features(
        df_train,
        feature_name='value',
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29, 30, 364, 365, 366]
    )

    validation_days_for_split = 30

    df_train_adjusted, df_val = create_val_from_train_silent(df_train_processed, validation_days_for_split)

    lag_columns = [
        'value_lag_1', 'value_lag_2', 'value_lag_3', 'value_lag_4', 'value_lag_5',
        'value_lag_6', 'value_lag_7', 'value_lag_8',
        'value_lag_13', 'value_lag_14', 'value_lag_15',
        'value_lag_20', 'value_lag_21', 'value_lag_22',
        'value_lag_27', 'value_lag_28', 'value_lag_29', 'value_lag_30',
        'value_lag_364', 'value_lag_365', 'value_lag_366'
    ]

    max_lag_window = 366

    grouping_keys = ['item_id', 'store_id'] 

    df_test_prepared = df_test.copy()
    for col in lag_columns:
        df_test_prepared[col] = np.nan 

    historical_context = df_train.groupby(grouping_keys).tail(max_lag_window)[['day'] + grouping_keys + ['value']].copy()

    test_skeleton = df_test_prepared[['day'] + grouping_keys].copy()
    test_skeleton['value'] = np.nan 

    combined_for_lags_temp = pd.concat([historical_context, test_skeleton]).copy()

    for lag_name in lag_columns:
        lag_num = int(lag_name.split('_')[-1])
        combined_for_lags_temp[lag_name] = combined_for_lags_temp.groupby(grouping_keys)['value'].shift(lag_num)


    df_test_lags_calculated = combined_for_lags_temp[
        combined_for_lags_temp['day'] >= df_test_prepared['day'].min()
    ][['day'] + grouping_keys + lag_columns].copy()


    df_test_prepared = pd.merge(
        df_test_prepared,
        df_test_lags_calculated,
        on=['day'] + grouping_keys,
        how='left',
        suffixes=('', '_calculated')
    )

    for col in lag_columns:
        calculated_col_name = f'{col}_calculated'
        if calculated_col_name in df_test_prepared.columns:

            df_test_prepared[col] = df_test_prepared[calculated_col_name]
            df_test_prepared.drop(columns=[calculated_col_name], inplace=True)


    for col in lag_columns:
        df_test_prepared[col] = df_test_prepared[col].fillna(0)

    logger.info('############################ TREINO #############################')
    logger.info(df_train_adjusted)
    logger.info('############################ VALIDACAO #############################')
    logger.info(df_val)
    logger.info('############################ TESTE #############################')
    logger.info(df_test_prepared)

    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Salvar os DataFrames em formato Parquet
    logger.info("Salvando os dataframes processados em formato Parquet...")
    df_train_adjusted.to_parquet(os.path.join(output_dir, "train_data.parquet"))
    df_val.to_parquet(os.path.join(output_dir, "val_data.parquet"))
    df_test_prepared.to_parquet(os.path.join(output_dir, "test_data.parquet"))
    df_reorganized.to_parquet(os.path.join(output_dir, "df_full_original.parquet"))

    # 3. Salvar o grafo e o mapeamento de produtos (essenciais para o modelo)
    logger.info("Salvando o grafo e o mapeamento de produtos...")
    torch.save(graph_data, os.path.join(output_dir, "graph_data.pt"))
    joblib.dump(product_mapping, os.path.join(output_dir, "product_mapping.joblib"))
    
    logger.info("Processamento de dados concluído com sucesso!")

if __name__ == "__main__":
    main()