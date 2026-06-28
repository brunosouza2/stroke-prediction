# Stroke Prediction — Phase 2: Genetic Algorithm + LLM

**FIAP Pós-Tech IA para Devs — Tech Challenge Fase 2**

Otimização de modelos de previsão de AVC usando Algoritmos Genéticos, com integração de LLM para interpretação dos diagnósticos em linguagem natural.

## Contexto

Na Fase 1, treinamos dois modelos para prever risco de AVC:

| Modelo | Accuracy | Recall | F1 |
|--------|----------|--------|----|
| Regressão Logística | 83.37% | **44.00%** | 0.21 |
| Random Forest | 92.86% | 24.00% | 0.25 |

O **Recall** é a métrica principal (contexto médico: minimizar casos não detectados).

Nesta fase, aplicamos Algoritmos Genéticos para otimizar os hiperparâmetros e integramos um LLM para interpretar os resultados.

---

## Estrutura do Repositório

```
stroke-prediction-phase2/
├── data/
│   └── download_data.py          # Script para baixar o dataset do Kaggle
├── src/
│   ├── preprocessing.py          # Pipeline de pré-processamento (Fase 1 refatorada)
│   ├── models.py                 # Treinamento, avaliação, save/load, predict
│   ├── genetic_algorithm/        # Motor do AG (Integrante 1)
│   └── llm/                      # Integração LLM (Integrante 2)
├── notebooks/
│   ├── 01_baseline.ipynb         # Reprodução da Fase 1 (linha de base)
│   ├── 02_genetic_algorithm.ipynb# Experimentos com AG
│   └── 03_llm_integration.ipynb  # Demonstração da integração LLM
├── results/
│   ├── logistic_regression.joblib# Modelo LR treinado
│   ├── random_forest.joblib      # Modelo RF treinado
│   └── baseline_metrics.json     # Métricas da linha de base
├── tests/
├── docs/
├── .env.example
└── requirements.txt
```

---

## Pré-requisitos

- Python 3.10+
- Conta no Kaggle (para baixar o dataset)
- API key do Google Gemini (Integrante 2 — LLM integration)

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repo>
cd stroke-prediction-phase2

# 2. Crie e ative um ambiente virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure variáveis de ambiente (Integrante 2)
copy .env.example .env
# Edite .env e adicione sua GOOGLE_API_KEY
```

---

## Dataset

O dataset utilizado é o [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) do Kaggle.

```bash
python data/download_data.py
```

Isso salva o CSV em `data/healthcare-dataset-stroke-data.csv`.

> O CSV não está no repositório (arquivo grande). Rode o script acima após clonar.

---

## Como Executar

### 1. Reproduzir a linha de base (Fase 1)

```bash
jupyter notebook notebooks/01_baseline.ipynb
```

Isso treina os dois modelos baseline e salva em `results/`.

> **Integrante 2:** execute este notebook primeiro para gerar os `.joblib` necessários.

### 2. Experimentos com Algoritmo Genético

```bash
jupyter notebook notebooks/02_genetic_algorithm.ipynb
```

### 3. Demonstração LLM

```bash
jupyter notebook notebooks/03_llm_integration.ipynb
```

---

## Usando os módulos em Python

```python
# Carregar modelo treinado (usado pelo Integrante 2)
from src.models import load_model, predict
from src.preprocessing import prepare_pipeline

_, X_test, _, y_test = prepare_pipeline()
model = load_model("logistic_regression")
predictions, probabilities = predict(model, X_test)

# Rodar pipeline completo do zero
from src.preprocessing import prepare_pipeline
X_train, X_test, y_train, y_test = prepare_pipeline()
```

---

## Integrantes

| Papel | Responsabilidade |
|-------|-----------------|
| Integrante 1 | Algoritmo Genético (TASK-1.1 a 1.6) |
| Integrante 2 | Integração LLM (TASK-2.1 a 2.5) |
| Integrante 3 | Integração, testes, docs e relatório final (TASK-3.1 a 3.5) |

---

## Dependências Principais

| Biblioteca | Uso |
|-----------|-----|
| scikit-learn | Modelos de ML |
| imbalanced-learn | SMOTE |
| deap | Algoritmo Genético |
| google-generativeai | LLM (Gemini) |
| pandas / numpy | Manipulação de dados |
| joblib | Persistência de modelos |
| pytest | Testes automatizados |
