#!/bin/bash

echo "Iniciando processamento de dados..."
python data_processing.py

if [ $? -eq 0 ]; then
    echo "Processamento de dados concluído com sucesso. Iniciando o treino do modelo..."
    python train.py
    echo "Treino do modelo concluído."
else
    echo "Erro no processamento de dados. Abortando o treino."
    exit 1
fi