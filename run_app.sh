#!/bin/bash

# Script para executar o aplicativo Streamlit de detecção de anomalias

echo "🚀 Iniciando aplicativo de detecção de anomalias..."
echo "📁 Diretório: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "📦 Streamlit: $(streamlit --version)"
echo "🖥️  Dispositivo: CPU (otimizado para Streamlit Cloud)"

# Verifica se o arquivo app.py existe
if [ ! -f "app.py" ]; then
    echo "❌ Erro: arquivo app.py não encontrado!"
    exit 1
fi

# Verifica se o arquivo model_utils.py existe
if [ ! -f "model_utils.py" ]; then
    echo "❌ Erro: arquivo model_utils.py não encontrado!"
    exit 1
fi

# Verifica se a pasta models existe
if [ ! -d "models" ]; then
    echo "❌ Erro: pasta models não encontrada!"
    exit 1
fi

# Verifica se o modelo existe
if [ ! -f "models/bottle_unet_best.pth" ]; then
    echo "⚠️  Aviso: modelo bottle_unet_best.pth não encontrado!"
    echo "   Certifique-se de que o modelo está na pasta models/"
fi

# Verifica se a configuração existe
if [ ! -f "models/bottle_unet_config.json" ]; then
    echo "⚠️  Aviso: arquivo de configuração bottle_unet_config.json não encontrado!"
    echo "   Criando configuração padrão..."
    mkdir -p models
    echo '{"classification_threshold": 0.01, "pixel_visualization_threshold": 0.5}' > models/bottle_unet_config.json
fi

# Cria pasta de imagens se não existir
if [ ! -d "imagem" ]; then
    echo "📁 Criando pasta de imagens de exemplo..."
    mkdir -p imagem
fi

echo "✅ Verificações concluídas!"
echo "🌐 Iniciando servidor Streamlit..."
echo "📱 Acesse: http://localhost:8501"
echo ""

# Executa o Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
