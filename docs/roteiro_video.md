# Roteiro — Vídeo de Demonstração Tech Challenge Fase 2

**Duração total estimada: ~13 minutos**
Ferramenta sugerida: OBS Studio ou Loom para gravação de tela + áudio

---

## INTEGRANTE 1 — Partes 1 a 3 (~7 minutos)

---

### PARTE 1 — Introdução e Contexto (~2 min)

**Tela:** `README.md` no GitHub → `notebooks/01_baseline.ipynb` (outputs já prontos)

> "Olá! Estamos apresentando o Tech Challenge da Fase 2 do curso Pós-Tech IA para Devs da FIAP.
> Neste projeto, partimos do trabalho da Fase 1: um modelo de previsão de risco de AVC treinado
> com o Stroke Prediction Dataset do Kaggle — 5.110 registros e 4,9% de casos positivos.
> O desafio desta fase foi otimizar esses modelos usando Algoritmos Genéticos e integrar um LLM
> para gerar explicações em linguagem natural dos resultados."

**Tela:** Mostrar outputs salvos do `01_baseline.ipynb` (não é necessário re-rodar)

> "Aqui vemos o notebook baseline da Fase 1. A Regressão Logística atingiu recall de 44% e
> o Random Forest, 24%. O recall é nossa métrica principal: num contexto médico, um falso
> negativo — AVC não detectado — tem consequências muito mais graves que um falso positivo.
> Nosso objetivo era manter ou superar esses valores com o Algoritmo Genético."

---

### PARTE 2 — Algoritmo Genético (~3 min)

**Tela:** `src/genetic_algorithm/chromosome.py`

> "Vamos ver a implementação. O ponto de partida é a codificação dos cromossomos.
> Cada indivíduo da população representa um conjunto de hiperparâmetros.
> Para a Regressão Logística, temos 4 genes: o valor de C, o solver, max_iter e class_weight.
> Para o Random Forest, 5 genes: n_estimators, max_depth, min_samples_split,
> min_samples_leaf e class_weight."

**Tela:** `src/genetic_algorithm/fitness.py`

> "A função fitness avalia cada cromossomo com validação cruzada estratificada de 2 folds,
> usando apenas os dados de treino para evitar vazamento. A fórmula é:
> `0.6 × recall + 0.4 × F1`, com uma penalização de 0.15 para recall abaixo de 30%."

**Tela:** `src/genetic_algorithm/operators.py`

> "Os operadores genéticos: seleção por torneio, crossover single-point e mutação gaussiana
> para genes contínuos e mutação categórica para genes discretos. Implementamos também
> elitismo — os melhores indivíduos sempre passam para a próxima geração."

**Tela:** `notebooks/02_genetic_algorithm.ipynb` — **RODAR as células ao vivo**

> ⚠️ **Atenção:** este notebook leva ~4 minutos para rodar os 3 experimentos.
> Sugestão: rodar antes da gravação e mostrar os outputs já prontos.

**Ao mostrar a tabela comparativa:**

> "Aqui vemos os 3 experimentos. EXP-01 com população de 20 e LR, EXP-02 com população de 50
> e LR, e EXP-03 com população de 30 e Random Forest. Cada experimento foi configurado com
> taxas de mutação e cruzamento diferentes."

**Ao mostrar o gráfico de evolução:**

> "Este gráfico mostra a curva de evolução do fitness ao longo das gerações.
> Os experimentos convergiram rapidamente — em torno de 4 gerações — devido ao early stopping
> com patience de 4 gerações sem melhora significativa."

---

### PARTE 3 — Comparativo de Performance (~2 min)

**Tela:** Célula da tabela comparativa no `02_genetic_algorithm.ipynb`

> "Olhando os resultados: EXP-02 manteve recall de 44%, equivalente ao baseline da LR.
> O EXP-01 ficou em 42%. O EXP-03 com Random Forest ficou em 20% — abaixo do baseline de 24%.
> Isso nos mostrou um insight importante: o RF sem class_weight='balanced' prioriza accuracy
> no dataset desbalanceado, sacrificando recall."

**Tela:** Abrir `results/ga_summary.json` brevemente

> "Todos os resultados ficam persistidos em ga_summary.json — baseline, configurações de cada
> experimento, hiperparâmetros encontrados e métricas otimizadas. Isso garante rastreabilidade
> e reprodutibilidade completas."

> "Agora passo para o meu colega, que vai mostrar a integração com o LLM."

---

---

## INTEGRANTE 2 — Partes 4 a 6 (~6 minutos)

---

### PARTE 4 — Integração com LLM (~3 min)

**Tela:** `src/llm/` no VS Code — mostrar os 4 arquivos na barra lateral

> "A integração com LLM usa o Google Gemini via SDK oficial google-genai.
> A arquitetura tem 4 módulos: client.py gerencia a conexão com a API,
> prompts.py define os templates de prompt, interpreter.py orquestra as chamadas
> e evaluation.py avalia a qualidade das respostas sem custo de API."

**Tela:** `src/llm/prompts.py`

> "Temos dois prompts principais. O primeiro gera explicações de diagnósticos individuais:
> injeta as features do paciente como bullets, instrui o modelo a usar apenas os dados
> fornecidos e pede linguagem probabilística — nunca um diagnóstico categórico direto.
> O segundo prompt gera um resumo executivo do experimento com métricas e deltas."

**Tela:** `src/llm/evaluation.py`

> "Para avaliar a qualidade, implementamos um checklist determinístico de 4 regras:
> se o recall foi mencionado, se há valores numéricos dos dados no texto,
> se usa linguagem de risco e se o tamanho é razoável.
> O score é a média dessas 4 regras, de 0 a 1. Isso roda em CI sem custo de API."

**Tela:** `notebooks/03_llm_integration.ipynb` — **RODAR as células ao vivo**

> ⚠️ **Atenção:** Este notebook faz chamadas reais à API do Gemini.
> Configure a variável antes de gravar: `$env:GEMINI_API_KEY = "sua-chave-aqui"`

**Ao mostrar a saída do `explain_prediction()`:**

> "Aqui vemos o Gemini gerando uma explicação para um paciente específico.
> O modelo recebeu: idade, hipertensão, tipo de trabalho, nível de glicose e IMC.
> A resposta está em português e usa linguagem adequada ao contexto médico — note que
> ele fala em 'risco elevado', não em 'você tem AVC'."

**Ao mostrar o `summarize_experiment()`:**

> "E aqui o resumo executivo do experimento. O Gemini recebeu as métricas do AG e gerou
> um texto para gestores de saúde, explicando o que o delta de recall significa
> em termos práticos para a triagem de pacientes."

**Ao mostrar o score de avaliação:**

> "O score de qualidade deste exemplo mostra quais das 4 regras foram atendidas."

---

### PARTE 5 — Desafios e Soluções (~1.5 min)

**Tela:** `docs/relatorio_tecnico.pdf` seção 5 ou `docs/architecture.md` no GitHub

> "Alguns desafios que enfrentamos. O principal foi o custo computacional do AG:
> com CV de 5 folds e população de 50, cada experimento levava mais de 15 minutos.
> Resolvemos reduzindo para 2 folds e implementando early stopping agressivo —
> convergindo em ~2 minutos sem perda significativa de qualidade."

> "O dataset desbalanceado também foi um obstáculo: sem a penalização na fitness,
> o AG aprendia a maximizar accuracy ignorando recall. A penalização de 0.15 para
> recall < 30% resolveu isso."

> "Na integração com LLM, LLM-as-judge introduziria viés e custo. Optamos pelo
> checklist determinístico que é reprodutível em CI — uma decisão de trade-off consciente."

---

### PARTE 6 — Testes e Encerramento (~1.5 min)

**Tela:** Terminal — **RODAR os testes ao vivo**

```bash
cd stroke-prediction-phase2
pytest tests/ -v --tb=short
```

> "Para garantir robustez, implementamos uma suíte de 54 testes automatizados cobrindo
> todos os módulos: cromossomos, fitness, operadores, GA completo e todos os módulos LLM.
> Vamos rodar ao vivo."

**Após os testes passarem:**

> "54 testes passando — pipeline completo verificado."

**Tela:** Repositório no GitHub — mostrar commits, estrutura de pastas e PR

> "O código completo está disponível no repositório GitHub com toda a documentação,
> arquitetura em Mermaid e o relatório técnico em PDF e DOCX. Obrigado!"

---

## ✅ Checklist antes de gravar

### Ambos

- [ ] `git pull origin main`
- [ ] `pip install -r requirements.txt`
- [ ] Verificar resolução de tela (1080p recomendado) e áudio

### Integrante 1

- [ ] Rodar `notebooks/02_genetic_algorithm.ipynb` uma vez antes (ter outputs prontos como backup)
- [ ] Confirmar que `results/ga_summary.json` e `results/experiments.json` existem

### Integrante 2

- [ ] Configurar `$env:GEMINI_API_KEY = "sua-chave-aqui"` antes de gravar
- [ ] Testar `notebooks/03_llm_integration.ipynb` com a API key configurada
- [ ] Confirmar que `pytest tests/ -v` retorna 54 testes passando
