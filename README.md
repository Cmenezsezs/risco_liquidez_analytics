# risco_liquidez_analytics
Risco de liquidez
# Predictive Analytics & Stress Testing para Risco de Liquidez

![Python](https://shields.io)
![Data Science](https://shields.io)
![Market Risk](https://shields.io)

## Visão Geral do Projeto

Este repositório contém um pipeline analítico *end-to-end* desenvolvido para simular, modelar e prever o **Fluxo de Caixa e Risco de Liquidez** de uma instituição financeira sob cenários macroeconômicos voláteis. 

O projeto aborda o ciclo completo de uma solução de dados financeiros: desde a ingestão e simulação estocástica de dados de alta dimensão até o *Tuning* de modelos de Machine Learning, simulação de cenários de crise (*Stress Testing*) e geração de KPIs gerenciais para tomadas de decisão estratégicas.

> **Nota de Compliance:** Para garantir a segurança institucional e a conformidade com a LGPD, o pipeline utiliza um gerador de dados sintéticos estruturado via matrizes estocásticas que mimetizam com precisão o comportamento de volatilidade e eventos de cauda do mercado real, sem expor dados confidenciais corporativos.


## Arquitetura do Pipeline Analítico

O código está estruturado em 5 camadas lógicas consecutivas:

1. **Engenharia de Dados Estocásticos:** Geração de séries temporais financeiras complexas combinando tendências de longo prazo, sazonalidades de mercado e choques macroeconômicos (Câmbio, Inflação e IPCA). Inclui a simulação de **Heterocedasticidade** e **Eventos de Cauda (*Fat Tails*)** utilizando uma distribuição *t-Student* com caudas pesadas.
2. **Feature Engineering:** Criação automatizada de variáveis preditoras temporais, aplicando técnicas de defasagem (*Lags*) e métricas móveis de volatilidade (desvio padrão e médias móveis).
3. **Modelagem Preditiva Avançada:** Divisão temporal dos dados (evitando *Data Leakage*) e treinamento de um regressor não-linear (*Random Forest Regressor*) robusto a cenários de alta incerteza.
4. **Stress Testing (Simulação de Crise):** Submissão do modelo treinado a um cenário de estresse macroeconômico severo (choque de +40% na taxa de juros e +30% no câmbio) para avaliar a resiliência do caixa.
5. **Business Intelligence & Data Storytelling:** Tradução das saídas matemáticas em indicadores financeiros vitais (Dias sob risco e exposição máxima de perda) acompanhados de uma visualização gráfica executiva para *stakeholders*.


## Principais Métricas e Resultados

O modelo foi projetado para otimizar a acurácia de projeções e fornecer alertas antecipados. No cenário simulado padrão, o pipeline entrega:

*   **Métricas de Validação:** Avaliação rigorosa através de `APE` (Erro Percentual Médio Absoluto) e `RMSE`.
*   **Geração de Valor de Negócio:** Redução de decisões incorretas em momentos de estresse financeiro através do mapeamento analítico do limite crítico de liquidez.


## Stack Tecnológico Utilizado

*   **Linguagem:** Python 3.8+
*   **Manipulação e Engenharia de Dados:** `pandas`, `numpy`
*   **Modelagem Preditiva e Machine Learning:** `scikit-learn` (Random Forest, Métricas de Erro)
*   **Visualização e Storytelling:** `matplotlib`, `seaborn`

## Como Executar o Projeto

1. Clone este repositório para sua máquina local:
   ```bash
   git clone https://github.com
   ```
2. Instale as dependências necessárias:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Execute o script principal do pipeline:
   ```bash
   python risco_liquidez_analytics.py
   ```

## Conceitos Estatísticos e Financeiros Aplicados

Este projeto demonstra a aplicação prática de competências de nível **Pleno** em Finanças Quantitativas:
*   **Distribuições de Caudas Pesadas (*Fat Tails*):** Modelagem de risco em cenários onde a distribuição Normal clássica subestima a ocorrência de eventos extremos de mercado.
*   **Backtesting de Séries Temporais:** Validação rigorosa de modelos respeitando a cronologia dos dados financeiros históricos.
*   **Stress Testing:** Prática mandatória de governança e regulação bancária para simulação de liquidez sob crises macroeconômicas agudas.
