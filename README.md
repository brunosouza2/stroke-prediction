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
│   └── download_data.py            # Script para baixar o dataset do Kaggle
├── src/
│   ├── preprocessing.py            # Pipeline de pré-processamento (Fase 1 refatorada)
│   ├── models.py                   # Treinamento, avaliação, save/load, predict
│   ├── genetic_algorithm/          # Motor do AG
│   └── llm/                        # Integração LLM
├── notebooks/
│   ├── 01_baseline.ipynb           # Reprodução da Fase 1 (linha de base)
│   ├── 02_genetic_algorithm.ipynb  # Experimentos com AG
│   └── 03_llm_integration.ipynb    # Demonstração da integração LLM
├── results/
│   ├── logistic_regression.joblib  # Modelo LR treinado
│   ├── random_forest.joblib        # Modelo RF treinado
│   └── baseline_metrics.json       # Métricas da linha de base
├── tests/
├── docs/
├── .env.example
└── requirements.txt
```

---

## Pré-requisitos

- Python 3.10+
- Conta no Kaggle (para baixar o dataset)
- API key do Google Gemini

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

# 4. Configure variáveis de ambiente
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
# Carregar modelo treinado e gerar predições
from src.models import load_model, predict
from src.preprocessing import prepare_pipeline

_, X_test, _, y_test = prepare_pipeline()
model = load_model("logistic_regression")
predictions, probabilities = predict(model, X_test)

# Rodar pipeline completo do zero
from src.preprocessing import prepare_pipeline
X_train, X_test, y_train, y_test = prepare_pipeline()
```

### Integração com LLM

O módulo `src/llm` usa a API do Google Gemini (`google-genai`) para gerar
explicações em linguagem natural a partir dos diagnósticos e dos resultados
do algoritmo genético. Configure `GOOGLE_API_KEY` no `.env` (copie
`.env.example`); Estamos utilizando o modelo `gemini-3.5-flash`.

```python
import json

from src.llm import evaluate_explanation, explain_prediction, summarize_experiment

# Explicação individual de uma predição
patient_data = X_test.iloc[0].to_dict()
prediction, probability = int(predictions[0]), float(probabilities[0])

explanation = explain_prediction(patient_data, prediction, probability)
quality = evaluate_explanation(explanation, patient_data)
print(explanation)
print(quality["score"], quality["checks"])

# Interpretação agregada de um experimento do algoritmo genético
ga_summary = json.load(open("results/ga_summary.json"))
exp = ga_summary["experiments"][0]

summary = summarize_experiment(
    baseline_metrics=ga_summary["baseline"][exp["model_type"]],
    optimized_metrics=exp["optimized_metrics"],
    best_params=exp["ga"]["best_params"],
)
```

**Avaliação de qualidade:** `evaluate_explanation(text, source_data)` roda
um checklist determinístico — sem chamar a API — com 4 regras: grounding
numérico nos dados de entrada, menção ao recall quando a métrica está
presente em `source_data`, uso de linguagem de risco/probabilidade (em vez
de afirmação categórica de diagnóstico) e tamanho razoável da resposta.
Retorna `{"score": float, "checks": {regra: bool}}`. Ver
`notebooks/03_llm_integration.ipynb` para uma demonstração completa,
incluindo as notas de prompt engineering usadas.

---

## Integrantes

| Papel | Responsabilidade |
|-------|-----------------|
| Integrante 1 | Algoritmo Genético (TASK-1.1 a 1.6) |
| Integrante 2 | Integração LLM (TASK-2.1 a 2.8) |
| Integrante 3 | Integração, testes, docs e relatório final (TASK-3.1 a 3.5) |

---

## Dependências Principais

| Biblioteca | Uso |
|-----------|-----|
| scikit-learn | Modelos de ML |
| imbalanced-learn | SMOTE |
| deap | Algoritmo Genético |
| google-genai | LLM (Gemini) |
| pandas / numpy | Manipulação de dados |
| joblib | Persistência de modelos |
| pytest | Testes automatizados |
