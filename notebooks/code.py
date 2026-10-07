import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Configuração de estilo visual para os plots corporativos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.titlesize': 12})

# 1. ENGENHARIA DE DADOS & SIMULAÇÃO DE ALTA DIMENSÃO (HIGH-DIMENSIONAL)
# =====================================================================
print("--- 1. Iniciando Pipeline de Ingestão e Engenharia de Dados ---")
np.random.seed(42)
datas = pd.date_range(start="2022-01-01", end="2026-12-31", freq="D")
n_days = len(datas)

# Simulando componentes macroeconômicos (Alta dimensão / Volatilidade)
taxa_juros = np.random.normal(loc=12.25, scale=0.5, size=n_days)
inflacao_ipca = np.random.normal(loc=4.5, scale=0.2, size=n_days)
indice_cambio = np.cumsum(np.random.normal(loc=0.001, scale=0.02, size=n_days)) + 5.0

# Injetando Heterocedasticidade e Efeitos de Cauda (Choques de Crise)
componente_erro = np.random.standard_t(df=3, size=n_days) * 15000  # Caudas pesadas (t-Student)
variacao_volatilidade = np.sin(np.linspace(0, 20, n_days)) * 10000
erro_heterocedastico = componente_erro + variacao_volatilidade

# Construindo a Série Temporal de Fluxo de Caixa (Receitas - Demandas de Liquidez)
tendencia = np.linspace(500000, 800000, n_days)
sazonalidade = np.sin(datas.dayofyear * (2 * np.pi / 365.25)) * 80000
fluxo_caixa = tendencia + sazonalidade - (indice_cambio * 5000) + erro_heterocedastico

df_financeiro = pd.DataFrame({
    'Data': datas,
    'Fluxo_Caixa_Real': fluxo_caixa,
    'Taxa_Juros': taxa_juros,
    'IPCA': inflacao_ipca,
    'Cambio': indice_cambio
}).set_index('Data')

# Feature Engineering (Lags e Médias Móveis para Séries Temporais)
for lag in:
    df_financeiro[f'Fluxo_Lag_{lag}'] = df_financeiro['Fluxo_Caixa_Real'].shift(lag)
df_financeiro['Media_Movel_7D'] = df_financeiro['Fluxo_Caixa_Real'].shift(1).rolling(window=7).mean()
df_financeiro['Volatilidade_Movel_30D'] = df_financeiro['Fluxo_Caixa_Real'].shift(1).rolling(window=30).std()

df_financeiro.dropna(inplace=True)
print(f"Dataset estruturado com {df_financeiro.shape[0]} registros e {df_financeiro.shape[1]} features.\n")

# 2. MODELAGEM PREDITIVA & TUNING (PREVISÃO DE ALTA PRECISÃO)
# =====================================================================
print("--- 2. Treinamento do Modelo de Machine Learning ---")
X = df_financeiro.drop(columns=['Fluxo_Caixa_Real'])
y = df_financeiro['Fluxo_Caixa_Real']

# Divisão Temporal (Crucial para Séries Temporais em Finanças)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Instanciando algoritmo robusto a não-linearidades (Random Forest)
modelo_risco = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
modelo_risco.fit(X_train, y_train)

# Backtesting e Métricas de Validação Analítica
previsoes = modelo_risco.predict(X_test)
mape = mean_absolute_percentage_error(y_test, previsoes)
rmse = np.sqrt(mean_squared_error(y_test, previsoes))

print(f"Métricas do Modelo no Dataset de Teste:")
print(f" -> MAPE (Erro Percentual Médio Absoluto): {mape:.2%}")
print(f" -> RMSE (Raiz do Erro Quadrático Médio): R$ {rmse:,.2f}\n")

# 3. SIMULAÇÃO DE CENÁRIOS DE CRISE & STRESS TESTING
# =====================================================================
print("--- 3. Rodando Stress Testing (Cenário de Crise de Liquidez) ---")
# Simulando um choque macroeconômico extremo: Juros disparam +40%, Câmbio estressa +30%
X_stress = X_test.copy()
X_stress['Taxa_Juros'] = X_stress['Taxa_Juros'] * 1.40
X_stress['Cambio'] = X_stress['Cambio'] * 1.30
X_stress['Volatilidade_Movel_30D'] = X_stress['Volatilidade_Movel_30D'] * 1.50 # Estresse na volatilidade

previsoes_crise = modelo_risco.predict(X_stress)

# 4. INTELIGÊNCIA DE NEGÓCIOS & KPIs DE LIQUIDEZ (BI)
# =====================================================================
print("--- 4. Calculando Métricas de BI e Geração de Insights ---")
df_resultados = pd.DataFrame({
    'Real': y_test,
    'Previsto_Base': previsoes,
    'Previsto_Estresse': previsoes_crise
}, index=X_test.index)

# KPIs de Risco Financeiro
limite_alerta_liquidez = 450000
dias_abaixo_limite_base = (df_resultados['Previsto_Base'] < limite_alerta_liquidez).sum()
dias_abaixo_limite_crise = (df_resultados['Previsto_Estresse'] < limite_alerta_liquidez).sum()
perda_maxima_projetada = df_resultados['Previsto_Base'].min() - df_resultados['Previsto_Estresse'].min()

print(f"KPIs Gerenciais para Tomada de Decisão:")
print(f" -> Dias sob Risco de Liquidez (Cenário Base): {dias_abaixo_limite_base} dias")
print(f" -> Dias sob Risco de Liquidez (Cenário de Crise): {dias_abaixo_limite_crise} dias")
print(f" -> Exposição Máxima ao Risco Adicional (Stress): R$ {perda_maxima_projetada:,.2f}\n")

# 5. DATA STORYTELLING (VISUALIZAÇÃO DE RESULTADOS EXECUTIVOS)
# =====================================================================
print("--- 5. Gerando Visualização Estratégica para Stakeholders ---")
# Criando um gráfico corporativo de duas séries temporais para análise de crossover
plt.figure(figsize=(12, 6))
plt.plot(df_resultados.index[-90:], df_resultados['Real'].tail(90), label='Fluxo de Caixa Real', color='#1f77b4', linewidth=1.5)
plt.plot(df_resultados.index[-90:], df_resultados['Previsto_Base'].tail(90), label='Previsão Cenário Base', color='#2ca02c', linestyle='--', linewidth=1.5)
plt.plot(df_resultados.index[-90:], df_resultados['Previsto_Estresse'].tail(90), label='Projeção Estressada (Choque Macro)', color='#d62728', linestyle=':', linewidth=2)

plt.axhline(y=limite_alerta_liquidez, color='black', linestyle='-.', alpha=0.7, label='Limite de Alerta de Liquidez')
plt.fill_between(df_resultados.index[-90:], df_resultados['Previsto_Estresse'].tail(90), limite_alerta_liquidez, 
                 where=(df_resultados['Previsto_Estresse'].tail(90) < limite_alerta_liquidez), 
                 color='#d62728', alpha=0.15, label='Região de Déficit Crítico')

plt.title('Backtesting de Séries Temporais e Stress Testing do Fluxo de Caixa (Últimos 90 Dias)')
plt.xlabel('Data')
plt.ylabel('Valor em Reais (R$)')
plt.legend(loc='lower left', frameon=True)
plt.tight_layout()

# O gráfico seria exibido ou salvo no pipeline automatizado
plt.show()
print("Pipeline executado com sucesso e relatórios consolidados.")

