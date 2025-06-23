# import pandas as pd
# import numpy as np

# age = 26
# height_m = 1.7
# num_days = 200  # total samples

# # Helper to generate features with noise and broad realistic ranges
# def generate_feature(mean, min_val, max_val, noise_level, size):
#     return np.clip(np.random.normal(mean, noise_level, size), min_val, max_val)

# # Generate features across broad ranges mixing low & high risk possibilities
# data = {
#     'fitness_age': generate_feature(26, 18, 40, 7, num_days),
#     'resting_heart_rate': generate_feature(60, 45, 75, 10, num_days),
#     'min_respiration': np.round(generate_feature(12, 6, 20, 4, num_days)),
#     'max_respiration': np.round(generate_feature(18, 12, 26, 4, num_days)),
#     'weight_kg': generate_feature(70, 53, 90, 10, num_days),
#     'avg_sleep_spo2': generate_feature(94, 85, 99, 5, num_days),
#     'avg_spo2': generate_feature(94, 85, 99, 5, num_days),
#     'sleep_avg_respiration': generate_feature(15, 10, 20, 3, num_days),
#     'sleep_resting_heart_rate': generate_feature(50, 30, 75, 15, num_days),
#     'sleep_time_sec': generate_feature(25200, 15000, 32000, 6000, num_days),
#     'steps': generate_feature(5000, 2000, 8000, 2000, num_days),
#     'hydration_ml': generate_feature(1500, 900, 2200, 400, num_days),
#     'max_heart_rate': generate_feature(175, 150, 190, 20, num_days),
#     'min_heart_rate': generate_feature(50, 30, 70, 10, num_days)
# }

# df = pd.DataFrame(data)

# # Inject some high steps outliers
# num_outliers = 10  # Adjust as needed
# outlier_indices = np.random.choice(df.index, size=num_outliers, replace=False)
# df.loc[outlier_indices, 'steps'] = np.random.randint(8000, 20001, size=num_outliers)

# # Dependent/calculated columns
# df['avg_waking_respiration'] = (df['min_respiration'] + df['max_respiration']) / 2
# df['bmi'] = df['weight_kg'] / (height_m ** 2)
# df['min_spo2'] = df['avg_sleep_spo2'] - np.random.uniform(2, 5, size=num_days)
# df['min_spo2'] = df['min_spo2'].clip(lower=80)
# df['vo2_max_precise'] = 55 - 0.5 * df['fitness_age'] - 0.1 * df['resting_heart_rate']
# df['vo2_max_precise'] = df['vo2_max_precise'].clip(lower=30, upper=60)

# # Scoring function
# def compute_score(row, age=26):
#     score = 0
#     weights = {
#         'vo2max_bmi': 0.13,
#         'vo2max_sleep_time': 0.09,
#         'bmi_resting_hr': 0.12,
#         'sleep_resting_hr': 0.08,
#         'sleep_avg_respiration': 0.07,
#         'avg_waking_respiration_resting_hr': 0.10,
#         'fitness_age_resting_hr': 0.11,
#         'avg_sleep_spo2_sleep_time': 0.08,
#         'steps': 0.06,
#         'hydration_ml': 0.04,
#         'avg_spo2': 0.07,
#         'avg_waking_respiration': 0.05
#     }

#     if row['vo2_max_precise'] < 41.7 and (row['bmi'] < 18.5 or row['bmi'] > 24.9):
#         score += weights['vo2max_bmi']
#     if row['vo2_max_precise'] < 41.7 and (row['sleep_time_sec'] < 21600 or row['sleep_time_sec'] > 28800):
#         score += weights['vo2max_sleep_time']
#     if (row['bmi'] < 18.5 or row['bmi'] > 24.9) and row['resting_heart_rate'] > 60:
#         score += weights['bmi_resting_hr']
#     if (row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20) and row['resting_heart_rate'] > 60:
#         score += weights['avg_waking_respiration_resting_hr']
#     if row['fitness_age'] > age and row['resting_heart_rate'] > 60:
#         score += weights['fitness_age_resting_hr']
#     if row['avg_sleep_spo2'] < 95.0 and (row['sleep_time_sec'] / 3600 < 6 or row['sleep_time_sec'] / 3600 > 9):
#         score += weights['avg_sleep_spo2_sleep_time']
#     if row['steps'] < 4500:
#         score += weights['steps']
#     if row['hydration_ml'] < 1500:
#         score += weights['hydration_ml']
#     if row['avg_spo2'] < 95.0:
#         score += weights['avg_spo2']
#     if (row['avg_waking_respiration'] < 12 or row['avg_waking_respiration'] > 20):
#         score += weights['avg_waking_respiration']
#     if row['sleep_resting_heart_rate'] < 40 or row['sleep_resting_heart_rate'] > 60:
#         score += weights['sleep_resting_hr']
#     if (row['sleep_avg_respiration'] < 14 or row['sleep_avg_respiration'] > 16):
#         score += weights['sleep_avg_respiration']

#     if score <= 0.3:
#         return 0
#     elif score >= 0.3 and score <= 0.6:
#         return 1
#     else:
#         return 2

# # Apply the risk score function
# df['cvd_risk'] = df.apply(compute_score, axis=1)

# # Format output
# round_0_decimal = ['avg_spo2', 'sleep_avg_respiration', 'avg_sleep_spo2', 'min_spo2']
# for col in round_0_decimal:
#     df[col] = df[col].round(0).astype(int)

# cols_to_round_int = df.columns.difference(['vo2_max_precise', 'weight_kg' ,'cvd_risk'])
# df[cols_to_round_int] = df[cols_to_round_int].round(0).astype(int)

# df['vo2_max_precise'] = df['vo2_max_precise'].round(1)
# df['weight_kg'] = df['weight_kg'].round(1)

# # Reorder columns
# new_order = [
#     'vo2_max_precise', 'steps', 'min_heart_rate', 'max_heart_rate', 'resting_heart_rate',
#     'max_respiration', 'min_respiration', 'avg_waking_respiration', 'weight_kg',
#     'fitness_age', 'bmi', 'hydration_ml', 'avg_spo2', 'min_spo2', 'avg_sleep_spo2',
#     'sleep_time_sec', 'sleep_avg_respiration', 'sleep_resting_heart_rate', 'cvd_risk'
# ]
# df = df[new_order]

# # Save and show counts
# df.to_csv('combined_risk_generated_data.csv', index=False)
# print(df['cvd_risk'].value_counts())

import json
import random
import time
import joblib
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import os
from sklearn.svm import SVC


option = input("What is your option Perfil-0, Perfil-my-0, Perfil-1, Perfil-my-1, Perfil-2, Perfil-my-2, concat, process, ml_A ? ")

match option:
    case "Perfil-0":

        # random.seed(42)
        # start_date = datetime.date(2024, 1, 1)
        # days = 200

        # data = []

        # # Função auxiliar para gerar uma data
        # def generate_dates(start_date, days):
        #     return [(start_date + datetime.timedelta(days=i)).isoformat() for i in range(days)]


        # for date in generate_dates(start_date, days):
        #     vo2 = round(random.uniform(44.6, 45.1), 1)
        #     steps = random.randint(8400, 9500)
        #     min_hr = random.randint(46, 49)
        #     max_hr = random.randint(173, 179)
        #     max_resp = random.randint(18, 20)
        #     min_resp = random.randint(10, 12)
        #     avg_wake_resp = round(random.randint(13, 15))
        #     weight = round(random.uniform(67.0, 67.2), 1)
        #     bmi = round(random.uniform(22.8, 22.9), 1)

        #     # resting heart rate depende do VO2 max
        #     base_rest_hr = 55 - (vo2 - 44.6) * 8  # Ajuste o fator 8 para intensidade da correlação
        #     rest_hr = int(round(base_rest_hr + random.gauss(0, 1)))  # ruído gaussiano com std=1

        #     # Normalizar BMI para escala 0-1
        #     bmi_min, bmi_max = 22.8, 22.9
        #     bmi_factor = (bmi - bmi_min) / (bmi_max - bmi_min)  # 0 a 1
            
        #     # Normalizar VO2max para escala 0-1
        #     vo2_min, vo2_max = 44.6, 45.1
        #     vo2_factor = (vo2 - vo2_min) / (vo2_max - vo2_min)  # 0 a 1
            
        #     fitness_age_base = 18
        #     # BMI aumenta fitness age, VO2max diminui
        #     fitness_age = max(fitness_age_base, fitness_age_base + int(round(bmi_factor)) - int(round(vo2_factor)))
            
        #     hydration = random.randint(1880, 2380)
        #     avg_spo2 = random.randint(97, 98)
        #     min_spo2 = random.randint(92, 94)
        #     avg_sleep_spo2 = avg_spo2 - 1 if random.random() > 0.5 else avg_spo2
        #     sleep_time = random.randint(25100, 27100)
        #     sleep_avg_resp = round(random.randint(13, 15))
        #     sleep_rest_hr = random.randint(51, 54)

        #     data.append({
        #         "calendar_date": date,
        #         "vo2_max_precise": vo2,
        #         "steps": steps,
        #         "min_heart_rate": min_hr,
        #         "max_heart_rate": max_hr,
        #         "resting_heart_rate": rest_hr,
        #         "max_respiration": max_resp,
        #         "min_respiration": min_resp,
        #         "avg_waking_respiration": avg_wake_resp,
        #         "weight_kg": weight,
        #         "fitness_age": fitness_age,
        #         "bmi": bmi,
        #         "hydration_ml": hydration,
        #         "avg_spo2": avg_spo2,
        #         "min_spo2": min_spo2,
        #         "avg_sleep_spo2": avg_sleep_spo2,
        #         "sleep_time_sec": sleep_time,
        #         "sleep_avg_respiration": sleep_avg_resp,
        #         "sleep_resting_heart_rate": sleep_rest_hr
        #     })

        # df = pd.DataFrame(data)
        # new_order = [
        #     'calendar_date','vo2_max_precise', 'steps', 'min_heart_rate', 'max_heart_rate', 'resting_heart_rate',
        #     'max_respiration', 'min_respiration', 'avg_waking_respiration', 'weight_kg',
        #     'fitness_age', 'bmi', 'hydration_ml', 'avg_spo2', 'min_spo2', 'avg_sleep_spo2',
        #     'sleep_time_sec', 'sleep_avg_respiration', 'sleep_resting_heart_rate'
        # ]
        # df = df[new_order]

        #correlaçoes
        # corr_bmi = df['fitness_age'].corr(df['bmi'])
        # corr_vo2 = df['fitness_age'].corr(df['vo2_max_precise'])
        # corr_rest_hr_vo2 = df['resting_heart_rate'].corr(df['vo2_max_precise'])
        # corr_bmi_weight = df['bmi'].corr(df['weight_kg'])

        # print(f"Correlação fitness_age x BMI: {corr_bmi:.3f}")
        # print(f"Correlação fitness_age x VO2 Max Precise: {corr_vo2:.3f}")
        # print(f"Correlação entre resting_heart_rate e vo2_max_precise: {corr_rest_hr_vo2:.3f}")
        # print(f"Correlação entre bmi e weight_kg: {corr_bmi_weight:.3f}")

        # df.to_csv("perfil0_pontoA_dados.csv", index=False)

        # df = pd.read_csv("perfil0_pontoA_dados.csv")
        # df['cvd_risk'] = 0

        # # numeric_vars = [
        # #     'vo2_max_precise', 'steps', 'min_heart_rate', 'max_heart_rate', 'resting_heart_rate',
        # #     'max_respiration', 'min_respiration', 'avg_waking_respiration', 'weight_kg',
        # #     'fitness_age', 'bmi', 'hydration_ml', 'avg_spo2', 'min_spo2', 'avg_sleep_spo2',
        # #     'sleep_time_sec', 'sleep_avg_respiration', 'sleep_resting_heart_rate'
        # # ]

        # df.to_csv("perfil0_pontoA_dados.csv", index=False)

        # Carregar o dataset
        df = pd.read_csv("perfil0_pontoA_dados.csv")

        # Remover coluna de data para análises puramente numéricas
        df_numeric = df.drop(columns=["calendar_date", "cvd_risk"])

        # Estilo dos gráficos
        sns.set_theme(style="whitegrid", palette="muted")

        # 1. Histogramas
        df_numeric.hist(bins=30, figsize=(20, 15), edgecolor='black')
        plt.suptitle("Distribuição das Métricas (Histograma)", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # 2. Boxplots (para identificação de outliers)
        # Selecionar apenas colunas numéricas
        df_numeric = df.select_dtypes(include='number')

        # Definir layout da figura (número de colunas e linhas baseado no nº de variáveis)
        num_vars = len(df_numeric.columns)
        cols = 4  # Número de colunas por linha
        rows = (num_vars + cols - 1) // cols  # Calcula número de linhas necessárias

        # Criar figura e eixos
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # Gerar boxplot para cada variável
        for i, col in enumerate(df_numeric.columns):
            ax = axes[i]
            sns.boxplot(data=df_numeric, x=col, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Boxplot - {col}")
            axes[i].set_xlabel("")
            axes[i].grid(True)

            # Evita múltiplas legendas repetidas
            if i == 0:
                ax.legend()

        # Apagar subplots não utilizados (caso nº de métricas < nº de subplots criados)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # 3. Heatmap de correlação
        # Calcula a matriz de correlação
        correlation_matrix = df_numeric.corr()

        # Cria uma máscara para a parte superior da matriz
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        # Define a figura
        plt.figure(figsize=(16, 12))

        # Cria o heatmap com a máscara
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75}
        )
        plt.title("Mapa de Calor – Correlação entre Variáveis", fontsize=20)
        plt.tight_layout()
        plt.show()
        
    case "Perfil-my-0":
        # # Base data from user profile
        # base_height = 175
        # base_age = 26
        # min_fitness_age = 18

        # # # Function to calculate fitness age based on VO2max and BMI
        # # def calculate_fitness_age(vo2max, bmi):
        # #     bmi_factor = max(0, (bmi - 21.5) * 1.8)  # mais peso se IMC > 21.5
        # #     vo2_factor = max(0, (vo2max - 41.7) * 0.8)
        # #     fitness_age = max(min_fitness_age, int(round(base_age + bmi_factor - vo2_factor)))
        # #     return fitness_age
        
        # def calculate_fitness_age(vo2max, bmi):
        #     # Penaliza apenas valores VO2 abaixo de 52 com força
        #     vo2_penalty = max(0, (52.0 - vo2max) * 1.0)  # penalização forte
        #     bmi_penalty = max(0, (bmi - 22.0) * 0.5)     # penalização leve para IMC > 22

        #     fitness_age = int(round(min_fitness_age + vo2_penalty + bmi_penalty))
            
        #     return min(fitness_age, 21)  # Limita ao máximo de 21

        # # Function to generate resting_heart_rate with negative correlation with VO2max
        # def calculate_resting_hr(vo2max):
        #     vo2_factor = (vo2max - 41.7) / (55.4 - 41.7)  # normaliza vo2max entre 0 e 1
        #     resting_hr = int(round(47 + (1 - vo2_factor) * 5))  # 47 a 52 conforme vo2_factor decresce
        #     resting_hr = max(47, min(52, resting_hr))  # limita entre 47 e 52
        #     return resting_hr

        # # Generate 200 days of data
        # start_date = datetime.date(2024, 1, 1)
        # data = []

        # for i in range(200):
        #     date = start_date + datetime.timedelta(days=i)
        #     steps = random.randint(5000, 7000) if i % 7 != 5 else random.randint(10000, 20000)
        #     vo2max = round(random.uniform(41.7, 55.4), 1)
        #     weight = round(random.uniform(65.5, 66.5), 1)
        #     bmi = round(weight / ((base_height / 100) ** 2), 1)
        #     resting_hr = calculate_resting_hr(vo2max)
        #     max_hr = random.randint(165, 178)
        #     min_hr = random.randint(40, 50)
        #     max_resp = random.randint(18, 24)
        #     min_resp = random.randint(10, 13)
        #     avg_waking_resp = round(random.randint(13, 17))
        #     hydration = random.randint(2200, 3200)
        #     avg_spo2 = random.randint(96, 99)
        #     min_spo2 = random.randint(90, 95)
        #     avg_sleep_spo2 = random.randint(min_spo2 + 2, 97)
        #     sleep_time = random.randint(6*3600, 8*3600 + 600)
        #     sleep_avg_resp = random.randint(11, 15)
        #     sleep_rest_hr = random.randint(45, 52)
        #     fitness_age = calculate_fitness_age(vo2max, bmi)

        #     data.append({
        #         "calendar_date": date.strftime('%Y-%m-%d'),
        #         "vo2_max_precise": vo2max,
        #         "steps": steps,
        #         "max_heart_rate": max_hr,
        #         "min_heart_rate": min_hr,
        #         "resting_heart_rate": resting_hr,
        #         "max_respiration": max_resp,
        #         "min_respiration": min_resp,
        #         "avg_waking_respiration": avg_waking_resp,
        #         "weight_kg": weight,
        #         "fitness_age": fitness_age,
        #         "bmi": bmi,
        #         "hydration_ml": hydration,
        #         "avg_spo2": avg_spo2,
        #         "min_spo2": min_spo2,
        #         "avg_sleep_spo2": avg_sleep_spo2,
        #         "sleep_time_sec": sleep_time,
        #         "sleep_avg_respiration": sleep_avg_resp,
        #         "sleep_resting_heart_rate": sleep_rest_hr
        #     })

        # df = pd.DataFrame(data)

        # new_order = [
        #     'calendar_date','vo2_max_precise', 'steps', 'min_heart_rate', 'max_heart_rate', 'resting_heart_rate',
        #     'max_respiration', 'min_respiration', 'avg_waking_respiration', 'weight_kg',
        #     'fitness_age', 'bmi', 'hydration_ml', 'avg_spo2', 'min_spo2', 'avg_sleep_spo2',
        #     'sleep_time_sec', 'sleep_avg_respiration', 'sleep_resting_heart_rate'
        # ]
        # df = df[new_order]

        # # Calcula correlações
        # corr_bmi = df['fitness_age'].corr(df['bmi'])
        # corr_vo2 = df['fitness_age'].corr(df['vo2_max_precise'])
        # corr_rest_hr_vo2 = df['resting_heart_rate'].corr(df['vo2_max_precise'])
        # corr_bmi_weight = df['bmi'].corr(df['weight_kg'])

        # print(f"Correlação fitness_age x BMI: {corr_bmi:.3f}")
        # print(f"Correlação fitness_age x VO2 Max Precise: {corr_vo2:.3f}")
        # print(f"Correlação entre resting_heart_rate e vo2_max_precise: {corr_rest_hr_vo2:.3f}")
        # print(f"Correlação entre bmi e weight_kg: {corr_bmi_weight:.3f}")

        # df.to_csv("perfil0_pontoB_dados.csv", index=False)

        
        df = pd.read_csv("perfil0_pontoB_dados.csv")
        df['cvd_risk'] = 0

        df.to_csv("perfil0_pontoB_dados.csv", index=False)

        # Carregar o dataset
        df = pd.read_csv("perfil0_pontoB_dados.csv")

        # Remover coluna de data para análises puramente numéricas
        df_numeric = df.drop(columns=["calendar_date", "cvd_risk"])

        # Estilo dos gráficos
        sns.set_theme(style="whitegrid", palette="muted")

        # 1. Histogramas
        df_numeric.hist(bins=30, figsize=(20, 15), edgecolor='black')
        plt.suptitle("Distribuição das Métricas (Histograma)", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # 2. Boxplots (para identificação de outliers)
        # Selecionar apenas colunas numéricas
        df_numeric = df.select_dtypes(include='number')

        # Definir layout da figura (número de colunas e linhas baseado no nº de variáveis)
        num_vars = len(df_numeric.columns)
        cols = 4  # Número de colunas por linha
        rows = (num_vars + cols - 1) // cols  # Calcula número de linhas necessárias

        # Criar figura e eixos
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # Gerar boxplot para cada variável
        for i, col in enumerate(df_numeric.columns):
            ax = axes[i]
            sns.boxplot(data=df_numeric, x=col, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Boxplot - {col}")
            axes[i].set_xlabel("")
            axes[i].grid(True)

            # Evita múltiplas legendas repetidas
            if i == 0:
                ax.legend()

        # Apagar subplots não utilizados (caso nº de métricas < nº de subplots criados)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # 3. Heatmap de correlação
        # Calcula a matriz de correlação
        correlation_matrix = df_numeric.corr()

        # Cria uma máscara para a parte superior da matriz
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        # Define a figura
        plt.figure(figsize=(16, 12))

        # Cria o heatmap com a máscara
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75}
        )
        plt.title("Mapa de Calor – Correlação entre Variáveis", fontsize=20)
        plt.tight_layout()
        plt.show()
    
    case "Perfil-1":

        # # Funções auxiliares
        # def generate_vo2max(days, exercise_days):
        #     vo2_values = []
        #     last_vo2 = round(random.uniform(37.0, 39.0), 1)
        #     for i in range(days):
        #         if i in exercise_days:
        #             last_vo2 += round(random.uniform(-0.4, 0.6), 1)
        #             last_vo2 = max(35.0, min(last_vo2, 45.0))
        #         vo2_values.append(round(last_vo2, 1))
        #     return vo2_values

        # def generate_fitness_age(vo2_list):
        #     ages = []
        #     for vo2 in vo2_list:
        #         if vo2 >= 39:
        #             ages.append(random.randint(22, 24))
        #         elif vo2 >= 38:
        #             ages.append(25)
        #         else:
        #             ages.append(26)
        #     return ages

        # def generate_weight(days, base_weight=79.5):
        #     return [round(base_weight + random.uniform(-0.3, 0.3), 1) for _ in range(days)]

        # def generate_bmi(weight_list, height_cm=171):
        #     height_m = height_cm / 100
        #     return [round(w / (height_m ** 2), 1) for w in weight_list]

        # def generate_heart_rates(days, exercise_days, vo2_list):
        #     max_hr = []
        #     resting_hr = []
        #     min_hr = []
        #     for i in range(days):
        #         if i in exercise_days:
        #             max_hr.append(random.randint(145, 160))
        #         else:
        #             max_hr.append(random.randint(110, 125))
        #         # Resting HR diminui conforme VO2 max aumenta
        #         rhr_base = 70 - (vo2_list[i] - 35) * 1.5  # ajuste o coeficiente conforme necessário
        #         rhr = int(round(rhr_base + random.gauss(0, 1)))  # ruído gaussiano
        #         resting_hr.append(rhr)
        #         min_hr.append(rhr - random.randint(3, 8))
        #     return max_hr, min_hr, resting_hr

        # def generate_hydration(days):
        #     return [random.randint(1300, 1900) + random.randint(-50, 49) for _ in range(days)]

        # def generate_respiration(days):
        #     max_resp, min_resp, avg_waking = [], [], []
        #     for _ in range(days):
        #         avg = random.randint(14, 16)
        #         avg_waking.append(avg)
        #         min_resp.append(avg - random.randint(1, 3))
        #         max_resp.append(avg + random.randint(1, 3))
        #     return max_resp, min_resp, avg_waking

        # def generate_sleep(days):
        #     sleep_sec = []
        #     avg_resp = []
        #     rest_hr = []
        #     for _ in range(days):
        #         duration = random.randint(27000, 33000)
        #         sleep_sec.append(duration)
        #         avg_resp.append(random.randint(13, 17))
        #         rest_hr.append(random.randint(60, 66))
        #     return sleep_sec, avg_resp, rest_hr

        # def generate_spo2(days):
        #     avg_spo2, min_spo2, sleep_spo2 = [], [], []
        #     for _ in range(days):
        #         min_s = round(random.uniform(91.0, 94.0), 1)
        #         avg_s = round(min_s + random.uniform(1.0, 3.0), 1)
        #         sleep_s = round(min_s + random.uniform(1.5, 2.5), 1)
        #         avg_spo2.append(avg_s)
        #         min_spo2.append(min_s)
        #         sleep_spo2.append(sleep_s)
        #     return avg_spo2, min_spo2, sleep_spo2

        # # Parâmetros principais
        # days = 200
        # start_date = datetime.date(2024, 5, 1)
        # dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
        # exercise_days = sorted(random.sample(range(days), 7))  # 2-3 vezes por semana

        # # Gerar dados
        # vo2 = generate_vo2max(days, exercise_days)
        # fitness_age = generate_fitness_age(vo2)
        # weight = generate_weight(days)
        # bmi = generate_bmi(weight)
        # max_hr, min_hr, rest_hr =  generate_heart_rates(days, exercise_days, vo2)
        # hydration = generate_hydration(days)
        # max_resp, min_resp, avg_waking = generate_respiration(days)
        # sleep_time, sleep_resp, sleep_rhr = generate_sleep(days)
        # avg_spo2, min_spo2, sleep_spo2 = generate_spo2(days)
        # steps = [random.randint(1800, 3700) for _ in range(days)]

        # # Compilar DataFrame
        # df = pd.DataFrame({
        #     "calendar_date": [d.strftime('%Y-%m-%d') for d in dates],
        #     "vo2_max_precise": vo2,
        #     "steps": steps,
        #     "max_heart_rate": max_hr,
        #     "min_heart_rate": min_hr,
        #     "resting_heart_rate": rest_hr,
        #     "max_respiration": max_resp,
        #     "min_respiration": min_resp,
        #     "avg_waking_respiration": avg_waking,
        #     "weight_kg": weight,
        #     "fitness_age": fitness_age,
        #     "bmi": bmi,
        #     "hydration_ml": hydration,
        #     "avg_spo2": avg_spo2,
        #     "min_spo2": min_spo2,
        #     "avg_sleep_spo2": sleep_spo2,
        #     "sleep_time_sec": sleep_time,
        #     "sleep_avg_respiration": sleep_resp,
        #     "sleep_resting_heart_rate": sleep_rhr
        # })

        # new_order = [
        #     'calendar_date','vo2_max_precise', 'steps', 'min_heart_rate', 'max_heart_rate', 'resting_heart_rate',
        #     'max_respiration', 'min_respiration', 'avg_waking_respiration', 'weight_kg',
        #     'fitness_age', 'bmi', 'hydration_ml', 'avg_spo2', 'min_spo2', 'avg_sleep_spo2',
        #     'sleep_time_sec', 'sleep_avg_respiration', 'sleep_resting_heart_rate'
        # ]
        # df = df[new_order]

        # corr_bmi = df['fitness_age'].corr(df['bmi'])
        # corr_vo2 = df['fitness_age'].corr(df['vo2_max_precise'])
        # corr_rest_hr_vo2 = df['resting_heart_rate'].corr(df['vo2_max_precise'])
        # corr_bmi_weight = df['bmi'].corr(df['weight_kg'])

        # print(f"Correlação fitness_age x BMI: {corr_bmi:.3f}")
        # print(f"Correlação fitness_age x VO2 Max Precise: {corr_vo2:.3f}")
        # print(f"Correlação entre resting_heart_rate e vo2_max_precise: {corr_rest_hr_vo2:.3f}")
        # print(f"Correlação entre bmi e weight_kg: {corr_bmi_weight:.3f}")

        # df.to_csv("perfil1_pontoA_dados.csv", index=False)

        # Carregar o dataset
        df = pd.read_csv("perfil1_pontoA_dados.csv")
        df['cvd_risk'] = 1

        df.to_csv("perfil1_pontoA_dados.csv", index=False)
        df = pd.read_csv("perfil1_pontoA_dados.csv")

        # Remover coluna de data para análises puramente numéricas
        df_numeric = df.drop(columns=["calendar_date", "cvd_risk"])

        # Estilo dos gráficos
        sns.set_theme(style="whitegrid", palette="muted")

        # 1. Histogramas
        df_numeric.hist(bins=30, figsize=(20, 15), edgecolor='black')
        plt.suptitle("Distribuição das Métricas (Histograma)", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # 2. Boxplots (para identificação de outliers)
        # Selecionar apenas colunas numéricas
        df_numeric = df.select_dtypes(include='number')

        # Definir layout da figura (número de colunas e linhas baseado no nº de variáveis)
        num_vars = len(df_numeric.columns)
        cols = 4  # Número de colunas por linha
        rows = (num_vars + cols - 1) // cols  # Calcula número de linhas necessárias

        # Criar figura e eixos
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # Gerar boxplot para cada variável
        for i, col in enumerate(df_numeric.columns):
            ax = axes[i]
            sns.boxplot(data=df_numeric, x=col, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Boxplot - {col}")
            axes[i].set_xlabel("")
            axes[i].grid(True)

            # Evita múltiplas legendas repetidas
            if i == 0:
                ax.legend()

        # Apagar subplots não utilizados (caso nº de métricas < nº de subplots criados)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # 3. Heatmap de correlação
        # Calcula a matriz de correlação
        correlation_matrix = df_numeric.corr()

        # Cria uma máscara para a parte superior da matriz
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        # Define a figura
        plt.figure(figsize=(16, 12))

        # Cria o heatmap com a máscara
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75}
        )
        plt.title("Mapa de Calor – Correlação entre Variáveis", fontsize=20)
        plt.tight_layout()
        plt.show()
         
    case "Perfil-my-1":

        # days = 200
        # data = []
        # base_date = datetime.date(2024,4,1)
        # vo2_base = 38  # ligeiramente abaixo do saudável
        # weight_range = (65.5, 67.5)
        # height_m = 1.71
        # exercise_days = sorted(random.sample(range(days), k=10))  # ~2-3 dias/semana

        # last_vo2 = vo2_base
        # vo2_list = []

        # def generate_fitness_age(vo2_list):
        #     ages = []
        #     for vo2 in vo2_list:
        #         if vo2 >= 39:
        #             ages.append(random.randint(22, 24))
        #         elif vo2 >= 38:
        #             ages.append(25)
        #         else:
        #             ages.append(26)
        #     return ages

        # for day in range(days):
        #     date = base_date + datetime.timedelta(days=day)
        #     weight = round(random.uniform(*weight_range), 1)
        #     bmi = round(weight / (height_m ** 2), 1)

        #     did_exercise = day in exercise_days
        #     if did_exercise:
        #         vo2_max = round(random.uniform(vo2_base, vo2_base + 1.2), 1)
        #         max_hr = random.randint(150, 160)
        #         steps = random.randint(3000, 5000)
        #     else:
        #         vo2_max = last_vo2
        #         max_hr = random.randint(110, 125)
        #         steps = random.randint(1500, 3000)

        #     last_vo2 = vo2_max
        #     vo2_list.append(vo2_max)

        #     min_hr = random.randint(50, 60)
        #     resting_hr = random.randint(63, 70)

        #     max_resp = random.randint(16, 19)
        #     min_resp = random.randint(11, 13)
        #     avg_resp = round(random.uniform(min_resp + 1, max_resp - 1))
        #     sleep_resp = round(random.uniform(13, 17))

        #     hydration = random.randint(1600, 2200)
        #     spo2_avg = round(random.uniform(96.0, 98.5), 1)
        #     spo2_min = round(random.uniform(91.0, 94.0), 1)
        #     spo2_sleep = round(random.uniform(spo2_min + 1.0, spo2_avg), 1)

        #     sleep_time = random.randint(27000, 34000)
        #     sleep_rest_hr = random.randint(60, 67)

        #     data.append({
        #         "calendar_date": date.strftime("%Y-%m-%d"),
        #         "vo2_max_precise": vo2_max,
        #         "steps": steps,
        #         "max_heart_rate": max_hr,
        #         "min_heart_rate": min_hr,
        #         "resting_heart_rate": resting_hr,
        #         "max_respiration": max_resp,
        #         "min_respiration": min_resp,
        #         "avg_waking_respiration": avg_resp,
        #         "weight_kg": weight,
        #         "bmi": bmi,
        #         "hydration_ml": hydration,
        #         "avg_spo2": spo2_avg,
        #         "min_spo2": spo2_min,
        #         "avg_sleep_spo2": spo2_sleep,
        #         "sleep_time_sec": sleep_time,
        #         "sleep_avg_respiration": sleep_resp,
        #         "sleep_resting_heart_rate": sleep_rest_hr
        #     })

        # fitness_ages = generate_fitness_age(vo2_list)
        # for i in range(len(data)):
        #     data[i]["fitness_age"] = fitness_ages[i]

        # df = pd.DataFrame(data)

        # new_order = [
        #     'calendar_date','vo2_max_precise', 'steps', 'min_heart_rate', 'max_heart_rate', 'resting_heart_rate',
        #     'max_respiration', 'min_respiration', 'avg_waking_respiration', 'weight_kg',
        #     'fitness_age', 'bmi', 'hydration_ml', 'avg_spo2', 'min_spo2', 'avg_sleep_spo2',
        #     'sleep_time_sec', 'sleep_avg_respiration', 'sleep_resting_heart_rate'
        # ]
        # df = df[new_order]

        # corr_bmi = df['fitness_age'].corr(df['bmi'])
        # corr_vo2 = df['fitness_age'].corr(df['vo2_max_precise'])
        # corr_rest_hr_vo2 = df['resting_heart_rate'].corr(df['vo2_max_precise'])
        # corr_bmi_weight = df['bmi'].corr(df['weight_kg'])

        # print(f"Correlação fitness_age x BMI: {corr_bmi:.3f}")
        # print(f"Correlação fitness_age x VO2 Max Precise: {corr_vo2:.3f}")
        # print(f"Correlação entre resting_heart_rate e vo2_max_precise: {corr_rest_hr_vo2:.3f}")
        # print(f"Correlação entre bmi e weight_kg: {corr_bmi_weight:.3f}")

        # df.to_csv("perfil1_pontoB_dados.csv", index=False)

        # Carregar o dataset
        df = pd.read_csv("perfil1_pontoB_dados.csv")
        df['cvd_risk'] = 1

        df.to_csv("perfil1_pontoB_dados.csv", index=False)
        df = pd.read_csv("perfil1_pontoB_dados.csv")

        # Remover coluna de data para análises puramente numéricas
        df_numeric = df.drop(columns=["calendar_date", "cvd_risk"])

        # Estilo dos gráficos
        sns.set_theme(style="whitegrid", palette="muted")

        # 1. Histogramas
        df_numeric.hist(bins=30, figsize=(20, 15), edgecolor='black')
        plt.suptitle("Distribuição das Métricas (Histograma)", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # 2. Boxplots (para identificação de outliers)
        # Selecionar apenas colunas numéricas
        df_numeric = df.select_dtypes(include='number')

        # Definir layout da figura (número de colunas e linhas baseado no nº de variáveis)
        num_vars = len(df_numeric.columns)
        cols = 4  # Número de colunas por linha
        rows = (num_vars + cols - 1) // cols  # Calcula número de linhas necessárias

        # Criar figura e eixos
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # Gerar boxplot para cada variável
        for i, col in enumerate(df_numeric.columns):
            ax = axes[i]
            sns.boxplot(data=df_numeric, x=col, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Boxplot - {col}")
            axes[i].set_xlabel("")
            axes[i].grid(True)

            # Evita múltiplas legendas repetidas
            if i == 0:
                ax.legend()

        # Apagar subplots não utilizados (caso nº de métricas < nº de subplots criados)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # 3. Heatmap de correlação
        # Calcula a matriz de correlação
        correlation_matrix = df_numeric.corr()

        # Cria uma máscara para a parte superior da matriz
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        # Define a figura
        plt.figure(figsize=(16, 12))

        # Cria o heatmap com a máscara
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75}
        )
        plt.title("Mapa de Calor – Correlação entre Variáveis", fontsize=20)
        plt.tight_layout()
        plt.show()

    case "Perfil-2":
        # np.random.seed(42)  # Para reprodutibilidade

        # # Configurações básicas
        # start_date = datetime.date(2024, 4, 1)
        # days = 200

        # def is_exercise_day(day_index):
        #     base = (day_index % 2 == 0)
        #     noise = np.random.rand() > 0.7
        #     return base or noise

        # # Inicializa listas
        # dates = []
        # vo2_max_list = []
        # steps_list = []
        # max_hr_list = []
        # min_hr_list = []
        # resting_hr_list = []
        # max_resp_list = []
        # min_resp_list = []
        # avg_wake_resp_list = []
        # weight_list = []
        # bmi_list = []
        # hydration_list = []
        # avg_spo2_list = []
        # min_spo2_list = []
        # avg_sleep_spo2_list = []
        # sleep_time_list = []
        # sleep_avg_resp_list = []
        # sleep_resting_hr_list = []
        # fitness_age_list = []

        # # Constantes
        # age = 26
        # weight_kg = 85
        # height_cm = 175
        # bmi = weight_kg / ((height_cm / 100) ** 2)
        # vo2max_low = 20.0
        # vo2max_high = 29.0

        # for i in range(days):
        #     date = start_date + datetime.timedelta(days=i)
        #     exercise = is_exercise_day(i)
            
        #     dates.append(date.strftime('%Y-%m-%d'))

        #     # VO2 max
        #     if i == 0:
        #         vo2_max = np.random.uniform(vo2max_low, vo2max_high)
        #     else:
        #         if exercise:
        #             vo2_max = np.clip(vo2_max_list[-1] + np.random.uniform(-1.5, 1.5), vo2max_low, vo2max_high)
        #         else:
        #             vo2_max = max(vo2max_low, vo2_max_list[-1] - np.random.uniform(0.1, 0.4))
        #     vo2_max = round(vo2_max, 1)
        #     vo2_max_list.append(vo2_max)

        #     # Adiciona dependência direta de fitness_age ao VO2 com menos ruído
        #     fitness_age_val = int(60 - (vo2_max * 0.7) + np.random.normal(0, 1.0))
        #     fitness_age_val = np.clip(fitness_age_val, 20, 65)

        #     fitness_age_list.append(fitness_age_val)

        #     # Passos
        #     steps = int(np.random.normal(3500 if exercise else 1500, 500))
        #     steps_list.append(max(0, steps))

        #     # HRs
        #     max_hr = int(np.random.normal(130 if exercise else 100, 8))
        #     max_hr_list.append(max_hr)

        #     min_hr = int(np.random.normal(55, 5))
        #     min_hr_list.append(min_hr)

        #     resting_hr_base = 78 - (vo2_max * 0.4)  # Ex: VO2 = 25 → HR ~ 68
        #     resting_hr = int(np.random.normal(resting_hr_base, 2))

        #     resting_hr_list.append(resting_hr)

        #     # Respiração
        #     max_resp = int(np.random.normal(20 if exercise else 18, 2))
        #     min_resp = int(np.random.normal(12, 1))
        #     avg_wake_resp = int(np.random.normal(16, 1))
        #     max_resp_list.append(max_resp)
        #     min_resp_list.append(min_resp)
        #     avg_wake_resp_list.append(avg_wake_resp)

        #     # Peso & BMI
        #     weight = round(weight_kg + np.random.normal(0, 0.5), 1)
        #     weight_list.append(weight)
        #     bmi_val = round(weight / ((height_cm / 100) ** 2), 1)
        #     bmi_list.append(bmi_val)

        #     # Hidratação
        #     hydration = int(np.random.normal(1650, 200))
        #     hydration_list.append(max(1000, hydration))

        #     # SpO2
        #     avg_spo2 = round(np.random.normal(95, 1), 1)
        #     min_spo2 = round(np.random.normal(90, 1.5), 1)
        #     avg_sleep_spo2 = round((avg_spo2 + min_spo2) / 2, 1)
        #     avg_spo2_list.append(avg_spo2)
        #     min_spo2_list.append(min_spo2)
        #     avg_sleep_spo2_list.append(avg_sleep_spo2)

        #     # Sono
        #     sleep_hours = np.random.choice(
        #         [np.random.uniform(4, 6), np.random.uniform(6, 8), np.random.uniform(8, 10)],
        #         p=[0.2, 0.6, 0.2]
        #     )
        #     sleep_time = int(sleep_hours * 3600)
        #     sleep_time_list.append(sleep_time)

        #     sleep_avg_resp = int(np.random.normal(15, 1))
        #     sleep_avg_resp_list.append(sleep_avg_resp)

        #     sleep_resting_hr = int(np.random.normal(resting_hr - 5, 2))
        #     sleep_resting_hr_list.append(sleep_resting_hr)

        # # Criar DataFrame
        # df = pd.DataFrame({
        #     "calendar_date": dates,
        #     "vo2_max_precise": vo2_max_list,
        #     "steps": steps_list,
        #     "max_heart_rate": max_hr_list,
        #     "min_heart_rate": min_hr_list,
        #     "resting_heart_rate": resting_hr_list,
        #     "max_respiration": max_resp_list,
        #     "min_respiration": min_resp_list,
        #     "avg_waking_respiration": avg_wake_resp_list,
        #     "weight_kg": weight_list,
        #     "bmi": bmi_list,
        #     "hydration_ml": hydration_list,
        #     "avg_spo2": avg_spo2_list,
        #     "min_spo2": min_spo2_list,
        #     "avg_sleep_spo2": avg_sleep_spo2_list,
        #     "sleep_time_sec": sleep_time_list,
        #     "sleep_avg_respiration": sleep_avg_resp_list,
        #     "sleep_resting_heart_rate": sleep_resting_hr_list,
        #     "fitness_age": fitness_age_list
        # })

        # new_order = [
        #     'calendar_date','vo2_max_precise', 'steps', 'min_heart_rate', 'max_heart_rate', 'resting_heart_rate',
        #     'max_respiration', 'min_respiration', 'avg_waking_respiration', 'weight_kg',
        #     'fitness_age', 'bmi', 'hydration_ml', 'avg_spo2', 'min_spo2', 'avg_sleep_spo2',
        #     'sleep_time_sec', 'sleep_avg_respiration', 'sleep_resting_heart_rate'
        # ]
        # df = df[new_order]

        # corr_bmi = df['fitness_age'].corr(df['bmi'])
        # corr_vo2 = df['fitness_age'].corr(df['vo2_max_precise'])
        # corr_rest_hr_vo2 = df['resting_heart_rate'].corr(df['vo2_max_precise'])
        # corr_bmi_weight = df['bmi'].corr(df['weight_kg'])

        # print(f"Correlação fitness_age x BMI: {corr_bmi:.3f}")
        # print(f"Correlação fitness_age x VO2 Max Precise: {corr_vo2:.3f}")
        # print(f"Correlação entre resting_heart_rate e vo2_max_precise: {corr_rest_hr_vo2:.3f}")
        # print(f"Correlação entre bmi e weight_kg: {corr_bmi_weight:.3f}")

        # # Exportar para CSV
        # df.to_csv("perfil2_pontoA_dados.csv", index=False)

        # Carregar o dataset
        #df = pd.read_csv("perfil2_pontoA_dados.csv")
        #df['cvd_risk'] = 2

        #df.to_csv("perfil2_pontoA_dados.csv", index=False)
        df = pd.read_csv("perfil2_pontoA_dados.csv")

        # Remover coluna de data para análises puramente numéricas
        df_numeric = df.drop(columns=["calendar_date", "cvd_risk"])

        # Estilo dos gráficos
        sns.set_theme(style="whitegrid", palette="muted")

        # 1. Histogramas
        df_numeric.hist(bins=30, figsize=(20, 15), edgecolor='black')
        plt.suptitle("Distribuição das Métricas (Histograma)", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # 2. Boxplots (para identificação de outliers)
        # Selecionar apenas colunas numéricas
        df_numeric = df.select_dtypes(include='number')

        # Definir layout da figura (número de colunas e linhas baseado no nº de variáveis)
        num_vars = len(df_numeric.columns)
        cols = 4  # Número de colunas por linha
        rows = (num_vars + cols - 1) // cols  # Calcula número de linhas necessárias

        # Criar figura e eixos
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # Gerar boxplot para cada variável
        for i, col in enumerate(df_numeric.columns):
            ax = axes[i]
            sns.boxplot(data=df_numeric, x=col, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Boxplot - {col}")
            axes[i].set_xlabel("")
            axes[i].grid(True)

            # Evita múltiplas legendas repetidas
            if i == 0:
                ax.legend()

        # Apagar subplots não utilizados (caso nº de métricas < nº de subplots criados)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # 3. Heatmap de correlação
        # Calcula a matriz de correlação
        correlation_matrix = df_numeric.corr()

        # Cria uma máscara para a parte superior da matriz
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        # Define a figura
        plt.figure(figsize=(16, 12))

        # Cria o heatmap com a máscara
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75}
        )
        plt.title("Mapa de Calor - Correlação entre Variáveis", fontsize=20)
        plt.tight_layout()
        plt.show()

    case "Perfil-my-2":

        # # Seed para reprodutibilidade
        # np.random.seed(42)

        # # Configurações básicas
        # start_date = datetime.date(2024, 4, 1)
        # days = 200

        # # Função para determinar se há exercício
        # def is_exercise_day(day_index):
        #     base = (day_index % 3 == 0)
        #     noise = np.random.rand() > 0.85
        #     return base or noise

        # # Inicialização de listas
        # dates, vo2_max_list, steps_list, max_hr_list, min_hr_list, resting_hr_list = [], [], [], [], [], []
        # max_resp_list, min_resp_list, avg_wake_resp_list = [], [], []
        # weight_list, bmi_list, hydration_list = [], [], []
        # avg_spo2_list, min_spo2_list, avg_sleep_spo2_list = [], [], []
        # sleep_time_list, sleep_avg_resp_list, sleep_resting_hr_list, fitness_age_list = [], [], [], []

        # # Constantes
        # age = 26
        # height_cm = 175
        # weight_kg = 85
        # bmi = weight_kg / ((height_cm / 100) ** 2)
        # vo2max_low = 17.0
        # vo2max_high = 24.0
        # fitness_age_base = 50

        # for i in range(days):
        #     date = start_date + datetime.timedelta(days=i)
        #     exercise = is_exercise_day(i)
        #     dates.append(date.strftime('%Y-%m-%d'))

        #    # VO2 max
        #     if i == 0:
        #         vo2_max = np.random.uniform(vo2max_low, vo2max_high)
        #     else:
        #         # Se não houve exercício, chance maior de queda de VO2 máx
        #         if exercise:
        #             delta = np.random.uniform(-1.0, 1.0)
        #         else:
        #             delta = np.random.uniform(-0.5, 0.0) if np.random.rand() < 0.5 else 0  # pequenas quedas
        #         vo2_max = np.clip(vo2_max_list[-1] + delta, vo2max_low, vo2max_high)

        #         # Simular quedas ocasionais agudas (ex: fadiga/doença leve)
        #         if np.random.rand() < 0.05:
        #             vo2_max -= np.random.uniform(0.5, 1.2)
        #     vo2_max_list.append(round(vo2_max, 1))


        #     # Passos
        #     steps = int(np.random.normal(2500 if exercise else 1000, 400))
        #     steps_list.append(max(0, steps))

        #     # Frequência cardíaca
        #     max_hr = int(np.random.normal(125 if exercise else 105, 10))
        #     min_hr = int(np.random.normal(58, 6))
        #     resting_hr = int(np.random.normal(75 if not exercise else 72, 4))
        #     max_hr_list.append(max_hr)
        #     min_hr_list.append(min_hr)
        #     resting_hr_list.append(resting_hr)

        #     # Respiração
        #     max_resp = int(np.random.normal(22 if exercise else 20, 2))
        #     min_resp = int(np.random.normal(13, 1))
        #     avg_wake_resp = int(np.random.normal(17, 1))
        #     max_resp_list.append(max_resp)
        #     min_resp_list.append(min_resp)
        #     avg_wake_resp_list.append(avg_wake_resp)

        #     # Peso e BMI
        #     weight = round(weight_kg + np.random.normal(0, 0.7), 1)
        #     weight_list.append(weight)
        #     bmi_val = round(weight / ((height_cm / 100) ** 2), 1)
        #     bmi_list.append(bmi_val)

        #     # Hidratação
        #     hydration = int(np.random.normal(1500, 250))
        #     hydration_list.append(max(800, hydration))

        #     # Saturação O2
        #     avg_spo2 = round(np.random.normal(94, 1.2), 1)
        #     min_spo2 = round(np.random.normal(88.5, 2), 1)
        #     avg_sleep_spo2 = round((avg_spo2 + min_spo2) / 2, 1)
        #     avg_spo2_list.append(avg_spo2)
        #     min_spo2_list.append(min_spo2)
        #     avg_sleep_spo2_list.append(avg_sleep_spo2)

        #     # Sono
        #     sleep_hours = np.random.choice(
        #         [np.random.uniform(4, 6), np.random.uniform(6, 8), np.random.uniform(2, 4)],
        #         p=[0.5, 0.3, 0.2]
        #     )
        #     sleep_time = int(sleep_hours * 3600)
        #     sleep_time_list.append(sleep_time)
        #     # Simulação de respiração anormal no sono
        #     if np.random.rand() < 0.1:
        #         # Apneia leve (respiração baixa)
        #         sleep_avg_resp = int(np.random.uniform(10, 13))
        #     elif np.random.rand() < 0.1:
        #         # Hiperventilação leve (respiração alta)
        #         sleep_avg_resp = int(np.random.uniform(20, 23))
        #     else:
        #         sleep_avg_resp = int(np.random.normal(16, 1.5))
        #     sleep_avg_resp_list.append(sleep_avg_resp)
        #     sleep_resting_hr = int(np.random.normal(resting_hr - 3, 2))
        #     sleep_resting_hr_list.append(sleep_resting_hr)

        #     # Fitness age
        #     if exercise:
        #         fitness_age_val = fitness_age_list[-1] - np.random.randint(0, 2) if fitness_age_list else fitness_age_base
        #     else:
        #         fitness_age_val = fitness_age_list[-1] + np.random.randint(0, 2) if fitness_age_list else fitness_age_base
        #     fitness_age_val = np.clip(fitness_age_val, 45, 60)
        #     fitness_age_list.append(fitness_age_val)

        # # Criar DataFrame
        # df = pd.DataFrame({
        #     "calendar_date": dates,
        #     "vo2_max_precise": vo2_max_list,
        #     "steps": steps_list,
        #     "max_heart_rate": max_hr_list,
        #     "min_heart_rate": min_hr_list,
        #     "resting_heart_rate": resting_hr_list,
        #     "max_respiration": max_resp_list,
        #     "min_respiration": min_resp_list,
        #     "avg_waking_respiration": avg_wake_resp_list,
        #     "weight_kg": weight_list,
        #     "bmi": bmi_list,
        #     "hydration_ml": hydration_list,
        #     "avg_spo2": avg_spo2_list,
        #     "min_spo2": min_spo2_list,
        #     "avg_sleep_spo2": avg_sleep_spo2_list,
        #     "sleep_time_sec": sleep_time_list,
        #     "sleep_avg_respiration": sleep_avg_resp_list,
        #     "sleep_resting_heart_rate": sleep_resting_hr_list,
        #     "fitness_age": fitness_age_list
        # })

        # new_order = [
        #     'calendar_date','vo2_max_precise', 'steps', 'min_heart_rate', 'max_heart_rate', 'resting_heart_rate',
        #     'max_respiration', 'min_respiration', 'avg_waking_respiration', 'weight_kg',
        #     'fitness_age', 'bmi', 'hydration_ml', 'avg_spo2', 'min_spo2', 'avg_sleep_spo2',
        #     'sleep_time_sec', 'sleep_avg_respiration', 'sleep_resting_heart_rate'
        # ]
        # df = df[new_order]

        # corr_bmi = df['fitness_age'].corr(df['bmi'])
        # corr_vo2 = df['fitness_age'].corr(df['vo2_max_precise'])
        # corr_rest_hr_vo2 = df['resting_heart_rate'].corr(df['vo2_max_precise'])
        # corr_bmi_weight = df['bmi'].corr(df['weight_kg'])

        # print(f"Correlação fitness_age x BMI: {corr_bmi:.3f}")
        # print(f"Correlação fitness_age x VO2 Max Precise: {corr_vo2:.3f}")
        # print(f"Correlação entre resting_heart_rate e vo2_max_precise: {corr_rest_hr_vo2:.3f}")
        # print(f"Correlação entre bmi e weight_kg: {corr_bmi_weight:.3f}")

        # df.to_csv("perfil2_pontoB_dados.csv", index=False)

        # Carregar o dataset
        df = pd.read_csv("perfil2_pontoB_dados.csv")
        df['cvd_risk'] = 2

        df.to_csv("perfil2_pontoB_dados.csv", index=False)

        df = pd.read_csv("perfil2_pontoB_dados.csv")

        # Remover coluna de data para análises puramente numéricas
        df_numeric = df.drop(columns=["calendar_date", "cvd_risk"])

        # Estilo dos gráficos
        sns.set_theme(style="whitegrid", palette="muted")

        # 1. Histogramas
        df_numeric.hist(bins=30, figsize=(20, 15), edgecolor='black')
        plt.suptitle("Distribuição das Métricas (Histograma)", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # 2. Boxplots (para identificação de outliers)

        # Definir layout da figura (número de colunas e linhas baseado no nº de variáveis)
        num_vars = len(df_numeric.columns)
        cols = 4  # Número de colunas por linha
        rows = (num_vars + cols - 1) // cols  # Calcula número de linhas necessárias

        # Criar figura e eixos
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # Gerar boxplot para cada variável
        for i, col in enumerate(df_numeric.columns):
            ax = axes[i]
            sns.boxplot(data=df_numeric, x=col, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Boxplot - {col}")
            axes[i].set_xlabel("")
            axes[i].grid(True)

            # Evita múltiplas legendas repetidas
            if i == 0:
                ax.legend()

        # Apagar subplots não utilizados (caso nº de métricas < nº de subplots criados)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # 3. Heatmap de correlação
        # Calcula a matriz de correlação
        correlation_matrix = df_numeric.corr()

        # Cria uma máscara para a parte superior da matriz
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        # Define a figura
        plt.figure(figsize=(16, 12))

        #correlaçoes
        corr_bmi = df['fitness_age'].corr(df['bmi'])
        corr_vo2 = df['fitness_age'].corr(df['vo2_max_precise'])
        corr_rest_hr_vo2 = df['resting_heart_rate'].corr(df['vo2_max_precise'])
        corr_bmi_weight = df['bmi'].corr(df['weight_kg'])

        print(f"Correlação fitness_age x BMI: {corr_bmi:.3f}")
        print(f"Correlação fitness_age x VO2 Max Precise: {corr_vo2:.3f}")
        print(f"Correlação entre resting_heart_rate e vo2_max_precise: {corr_rest_hr_vo2:.3f}")
        print(f"Correlação entre bmi e weight_kg: {corr_bmi_weight:.3f}")

        # Cria o heatmap com a máscara
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75}
        )
        plt.title("Mapa de Calor – Correlação entre Variáveis", fontsize=20)
        plt.tight_layout()
        plt.show()

    case "concat":

        def verificar_colunas(df_list, grupo_nome):
            colunas_ref = df_list[0].columns.tolist()
            todas_iguais = all(df.columns.tolist() == colunas_ref for df in df_list)

            print(f"[{grupo_nome}] Colunas consistentes: {todas_iguais}")
            
            if not todas_iguais:
                for i, df in enumerate(df_list):
                    print(f"Colunas do DataFrame {i}: {df.columns.tolist()}")
                raise ValueError(f"As colunas nos DataFrames do grupo '{grupo_nome}' não coincidem.")

        # Juntar perfis do Ponto A
        df_a0 = pd.read_csv("perfil0_pontoA_dados.csv")
        df_a1 = pd.read_csv("perfil1_pontoA_dados.csv")
        df_a2 = pd.read_csv("perfil2_pontoA_dados.csv")

        verificar_colunas([df_a0, df_a1, df_a2], "Ponto A")

        # Concatenar os 3 perfis (não relacionados com os teus dados)
        df_pontoA = pd.concat([df_a0, df_a1, df_a2], ignore_index=True)
        df_pontoA.to_csv("dados_consolidados_pontoA.csv", index=False)

        # Juntar perfis do Ponto B (relacionados com os teus dados)
        df_b0 = pd.read_csv("perfil0_pontoB_dados.csv")
        df_b1 = pd.read_csv("perfil1_pontoB_dados.csv")
        df_b2 = pd.read_csv("perfil2_pontoB_dados.csv")

        verificar_colunas([df_b0, df_b1, df_b2], "Ponto B")

        df_pontoB = pd.concat([df_b0, df_b1, df_b2], ignore_index=True)
        df_pontoB.to_csv("dados_consolidados_pontoB.csv", index=False)

    case "process":
        
        df = pd.read_csv("dados_consolidados_pontoB.csv")

        df_process = df.drop(columns=["calendar_date"])

        y_target = df_process['cvd_risk']

        x_features = df_process.drop(columns=['cvd_risk'])
        
        scaler = RobustScaler()
        x_scaled = pd.DataFrame(scaler.fit_transform(x_features), columns=x_features.columns)

        rf = RandomForestClassifier()
        
        rfe = RFE(estimator=rf)

        tscv = TimeSeriesSplit(n_splits=7)

        # Create the pipeline
        pipeline = Pipeline([
            ('feature_selection', rfe)
        ])
        # Define the grid parameters to search
        #sample weight not provided, so all samples have the same weight
        #monotonic_cst = None, since its not support for multioutput classifications (i.e. when n_outputs_ > 1)
        param_grid = {
            'feature_selection__n_features_to_select': [5, 6, 7, 8],
            'feature_selection__estimator__n_estimators': [100, 200, 300], # number of trees 100 to 500 is usually good. More = better accuracy, slower training.
            'feature_selection__estimator__max_depth': [3, 4, 5, 6], # None mean full grown. Controls overfitting and i wont use None since i have limited resources
            'feature_selection__estimator__min_samples_split': [6, 8, 10, 12],
            'feature_selection__estimator__min_samples_leaf': [4, 5, 6, 8, 10], # i kept the default value since altering the value may smooth the model, especially in regression.
            'feature_selection__estimator__max_features': ['sqrt', 'log2', 0.3, 0.5], #sqrt is default for classification, log2 may provide better results than sqrt but at a cost of performance
            'feature_selection__estimator__bootstrap': [True], # By setting true, it will not use the full dataset in each iteration
            'feature_selection__estimator__max_samples': [0.1, 0.3, 0.5], # 10%, 30% or 50% of total samples 
            'feature_selection__estimator__criterion': ['gini', 'entropy'], # https://datascience.stackexchange.com/questions/10228/when-should-i-use-gini-impurity-as-opposed-to-information-gain-entropy
            'feature_selection__estimator__class_weight': [None, 'balanced']
        }

        # Set up RandomSearchCV
        # given the number of parameters the n_iter reflects on the number of combinations presented around 3240. Example, by setting n_iter 10, it will randomly sample only 10 different combinations from those 3240 possible ones
        # i could use the max number of iteration to check all possible combinations but for that i would use gridsearch. SInce i have limited resources, i will use a reasonable number of iter to get a good enough combination
        # calculate the coverage given the number of iteration and the parameter grid
        
        random_search  = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=75, #increase if execution time ~10-20 minutes
            cv=tscv,
            scoring='accuracy',
            n_jobs=4, # leave one core available so that the VM doesnt crash instead of jobs=-1
            random_state=0
        )
        
        # Fit the model
        random_search.fit(x_scaled, y_target)
        
        best_rfe = random_search.best_estimator_['feature_selection']
        selected_features = x_scaled.columns[best_rfe.support_]

        print("Selected features:", selected_features.tolist())

        print("Best parameters found:", random_search.best_params_) # best hyperparameters found
        print("Best estimator found:", random_search.best_estimator_) # the full pipeline with best parameters
        print(f"Best CV accuracy: {random_search.best_score_:.3f}") # best CV score achieved
        #print("Best CV results:", random_search.cv_results_) # detailed info on all tried parameter combos (mean scores, timings, etc).

        pd.set_option('display.max_rows', None)
        results = pd.DataFrame(random_search.cv_results_)
        print(results[['mean_test_score', 'std_test_score']])

        # Plot feature importances
        rf_model = best_rfe.estimator_

        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importances from RFE with Random Forest')
        plt.tight_layout()
        plt.show()

        # Define the folder path for Point A results
        results_folder = 'results_point_B_2'

        # Check if folder exists, if not, create it
        os.makedirs(results_folder, exist_ok=True)


        selected_features_list = selected_features.tolist()
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        best_estimator = random_search.best_estimator_
        best_cv_results = random_search.cv_results_


        # Save prints/results to a text file
        with open(os.path.join(results_folder, 'random_search_results.txt'), 'w') as f:
            f.write(f"Selected features: {selected_features_list}\n")
            f.write(f"Best parameters found: {best_params}\n")
            f.write(f"Best CV score found: {best_score}\n")
            f.write(f"Best estimator found: {best_estimator}\n")
            f.write(f"Best cv rseults found: {best_cv_results}\n")

        # Save selected features as JSON
        with open(os.path.join(results_folder, 'selected_features.json'), 'w') as f:
            json.dump(selected_features_list, f)

        # Save the best model pipeline
        joblib.dump(best_estimator, os.path.join(results_folder, 'best_pipeline.pkl'))

        # Save feature importances as CSV
        importance_df.to_csv(os.path.join(results_folder, 'feature_importances.csv'), index=False)

    case "ml_A":
        # Load data
        df = pd.read_csv("dados_consolidados_pontoA.csv")
        df.drop(columns=["calendar_date"], inplace=True)

        # Load selected features
        with open("results_point_A/selected_features.json", "r") as f:
            selected_features = json.load(f)

        # Prepare data
        X = df[selected_features]
        y = df["cvd_risk"]

        # Scale features
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split data for final testing (ex., last 15% as test)
        test_size = int(len(X_scaled) * 0.15)
        X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]

        # SVM pipeline
        svm = SVC()
        pipeline = Pipeline([
            ('svm', svm)
        ])

        # Define hyperparameter search space
        param_dist = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['linear', 'rbf', 'poly'],
            'svm__gamma': ['scale', 'auto'],
            'svm__class_weight': [None, 'balanced']
        }

        # TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=8)

        # RandomizedSearchCV setup
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            scoring='accuracy',
            cv=tscv,
            n_jobs=4,
            random_state=42,
            verbose=1
        )

        # ⏱️ Track execution time
        start_time = time.time()

        # Fit the model
        random_search.fit(X_train, y_train)

        end_time = time.time()
        execution_time = end_time - start_time

        # Predict on test set
        y_pred = random_search.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix - Test Set")
        plt.tight_layout()

        # Save results
        results_folder = "svm_results_point_A"
        os.makedirs(results_folder, exist_ok=True)

        # Save classification report
        report_df.to_csv(os.path.join(results_folder, "classification_report.csv"))
        plt.savefig(os.path.join(results_folder, "confusion_matrix.png"))

        # Save model and metadata
        joblib.dump(random_search.best_estimator_, os.path.join(results_folder, "svm_best_model.pkl"))

        with open(os.path.join(results_folder, "training_info.txt"), "w") as f:
            f.write(f"Execution time (s): {execution_time:.2f}\n")
            f.write(f"Best parameters: {random_search.best_params_}\n")
            f.write(f"Best CV score: {random_search.best_score_:.3f}\n")

        print("✅ SVM training, evaluation, and saving completed.")