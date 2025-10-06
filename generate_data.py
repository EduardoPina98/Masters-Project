import json
import random
import time
import joblib
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cvd_risk_label
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

def annotate_boxplot(ax, series):
    stats = series.describe()
    q1 = stats['25%']
    q2 = stats['50%']
    q3 = stats['75%']
    min_val = stats['min']
    max_val = stats['max']

    data_range = max_val - min_val
    y_offset = max(data_range * 0.05, 0.2)

    # Adjust y-limits to give space for labels
    ax.set_ylim(min_val - y_offset * 2, max_val + y_offset * 2)

    # Label positions
    
    ax.text(0, q1, f"Q1: {q1:.2f}", ha='center', va='bottom', fontsize=9)

    # Determine if median overlaps with Q1 or Q3
    small_gap = y_offset * 1.5  # define what counts as 'too close'
    if abs(q2 - q1) < small_gap and abs(q3 - q2) < small_gap:
        # If too close on both sides, move median slightly above Q3
        ax.text(0, q3 + y_offset * 0.6, f"Med: {q2:.2f}", ha='center', va='bottom', fontsize=9, weight='bold')
    elif abs(q2 - q1) < abs(q3 - q2):
        # Closer to Q1 — shift upward
        ax.text(0, q2 + y_offset * 0.4, f"Med: {q2:.2f}", ha='center', va='bottom', fontsize=9, weight='bold')
    else:
        # Closer to Q3 — shift downward
        ax.text(0, q2 - y_offset * 0.4, f"Med: {q2:.2f}", ha='center', va='top', fontsize=9, weight='bold')

    ax.text(0, q3, f"Q3: {q3:.2f}", ha='center', va='bottom', fontsize=9)

metric_groups = {
    "Heart Rate Metrics": ['min_heart_rate', 'max_heart_rate', 'resting_heart_rate', 'sleep_resting_heart_rate'],
    "SPO2 Metrics": ['avg_spo2', 'min_spo2', 'avg_sleep_spo2'],
    "Respiration Metrics": ['max_respiration', 'min_respiration', 'avg_waking_respiration', 'sleep_avg_respiration'],
    "Body Composition": ['bmi', 'weight_kg'],
    "Fitness & Performance": ['vo2_max_precise', 'fitness_age'],
    "Sleep Metrics": ['sleep_time_sec'],
    "Movement & Hydration": ['steps', 'hydration_ml']
}

age = 26
height_cm = 175

option = input("Choose option: process, ml_A, ml_LR, test? ")

match option:
    case "process":
        
        df = pd.read_csv("realistic_cvd_dataset2.csv")

        df_process = df.drop(columns=["calendar_date", "cvd_risk", "weight_kg"])

        y_target = df_process['cvd_risk_numeric']

        x_features = df_process.drop(columns=['cvd_risk_numeric'])
        
        #Disadvanatage: This may not perform well when data is highly skewed.
        scaler = RobustScaler()
        x_scaled = pd.DataFrame(scaler.fit_transform(x_features, y_target), columns=x_features.columns)

        rf = RandomForestClassifier()
        
        rfe = RFE(estimator=rf)

        tscv = TimeSeriesSplit(n_splits=10)

        # Create the pipeline
        pipeline = Pipeline([
            ('feature_selection', rfe)
        ])
        # Define the grid parameters to search
        #monotonic_cst = None, since its not support for multioutput classifications (i.e. when n_outputs_ > 1)
        param_grid = {
            'feature_selection__n_features_to_select': [5, 6, 7, 8, 9],
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

        scoring = {
            'balanced_accuracy': 'balanced_accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'
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
            scoring=scoring,
            n_jobs=4, # leave one core available so that the VM doesnt crash instead of jobs=-1
            random_state=42,
            refit='f1_weighted'
        )

        start_time = time.time()
        # Fit the model
        random_search.fit(x_scaled, y_target)

        end_time = time.time()
        execution_time = end_time - start_time

        best_rfe = random_search.best_estimator_['feature_selection']
        selected_features = x_scaled.columns[best_rfe.support_]

        results = pd.DataFrame(random_search.cv_results_)
        best_index = random_search.best_index_

        best_results = results.loc[best_index, [
            'mean_test_balanced_accuracy', 'std_test_balanced_accuracy',
            'mean_test_f1_weighted', 'std_test_f1_weighted',
            'mean_test_precision_weighted', 'std_test_precision_weighted',
            'mean_test_recall_weighted', 'std_test_recall_weighted'
        ]]

        print(f"Best Scoring Metrics:{best_results}")

        print(f"Execution time: {execution_time/60:.2f} minutes)")
        print("Best parameters found:", random_search.best_params_) # best hyperparameters found
        print("Best estimator found:", random_search.best_estimator_) # the full pipeline with best parameters
        print(f"Best CV accuracy: {random_search.best_score_:.3f}") # best CV score achieved
        #print("Best CV results:", random_search.cv_results_) # detailed info on all tried parameter combos (mean scores, timings, etc).

        # pd.set_option('display.max_rows', None)
        # results = pd.DataFrame(random_search.cv_results_)
        # best_index = random_search.best_index_

        # best_results = results.loc[best_index, [
        #     'mean_test_accuracy', 'std_test_accuracy',
        #     'mean_test_f1_macro', 'std_test_f1_macro',
        #     'mean_test_balanced_accuracy', 'std_test_balanced_accuracy'
        # ]]

        # print(f"Best Scoring Metrics:{best_results}")


        # # Plot feature importances
        # rf_model = best_rfe.estimator_

        # importances = rf_model.feature_importances_
        # importance_df = pd.DataFrame({
        #     'Feature': selected_features,
        #     'Importance': importances
        # }).sort_values(by='Importance', ascending=False)

        # plt.figure(figsize=(10, 6))
        # sns.barplot(data=importance_df, x='Importance', y='Feature')
        # plt.title('Feature Importances from RFE with Random Forest')
        # plt.tight_layout()
        # plt.show()

        # # Define the folder path for Point A results
        # results_folder = 'results_point_B__100_10'

        # # Check if folder exists, if not, create it
        # os.makedirs(results_folder, exist_ok=True)


        # selected_features_list = selected_features.tolist()
        # best_params = random_search.best_params_
        # best_score = random_search.best_score_
        # best_estimator = random_search.best_estimator_
        # best_cv_results = random_search.cv_results_
        # # Add execution time as a column in the full results
        # best_cv_results['execution_time_sec'] = execution_time/60


        # # Save prints/results to a text file
        # with open(os.path.join(results_folder, 'random_search_results.txt'), 'w') as f:
        #     f.write(f"Selected features: {selected_features_list}\n")
        #     f.write(f"Best parameters found: {best_params}\n")
        #     f.write(f"Best CV score found: {best_score}\n")
        #     f.write(f"Best estimator found: {best_estimator}\n")
        #     f.write(f"Best cv results found: {best_cv_results}\n")

        # # Save selected features as JSON
        # with open(os.path.join(results_folder, 'selected_features.json'), 'w') as f:
        #     json.dump(selected_features_list, f)

        # # Save the best model pipeline
        # joblib.dump(best_estimator, os.path.join(results_folder, 'best_pipeline.pkl'))

        # # Save feature importances as CSV
        # importance_df.to_csv(os.path.join(results_folder, 'feature_importances.csv'), index=False)

        #best results (creating plots)
        #loaded_pipeline = joblib.load('/home/eduardo/Desktop/master_project/Masters-Project/results_point_A__25_5/best_pipeline.pkl')
        # loaded_pipeline = joblib.load('/home/eduardo/Desktop/master_project/Masters-Project/results_point_B__25_10/best_pipeline.pkl')


        # # Load your data (should be consistent with how you trained)
        # df = pd.read_csv("realistic_cvd_dataset.csv")
        # df_process = df.drop(columns=["calendar_date", "cvd_risk"])

        # y_target = df_process['cvd_risk_numeric']
        # x_features = df_process.drop(columns=['cvd_risk_numeric'])

        # scaler = RobustScaler()
        # x_scaled = pd.DataFrame(scaler.fit_transform(x_features, y_target), columns=x_features.columns)

        # # Re-initialize CV strategy
        # tscv = TimeSeriesSplit(n_splits=5)

        # # Get cross-validated predictions using the loaded pipeline
        # y_true_all = []
        # y_pred_all = []

        # for train_idx, test_idx in tscv.split(x_scaled):
        #     X_train, X_test = x_scaled.iloc[train_idx], x_scaled.iloc[test_idx]
        #     y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]

        #     loaded_pipeline.fit(X_train, y_train)
        #     y_pred = loaded_pipeline.predict(X_test)

        #     y_true_all.extend(y_test)
        #     y_pred_all.extend(y_pred)

        # # Evaluate after all folds
        # print(classification_report(y_true_all, y_pred_all))
        # print(confusion_matrix(y_true_all, y_pred_all))
        # #Confusion matrix and plot
        # conf_matrix = confusion_matrix(y_true_all, y_pred_all)
        # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        # disp.plot(cmap="Blues", values_format='d')
        # plt.title("Cross-Validated Confusion Matrix")
        # plt.show()

        # # Contar o número de registos por categoria (risco)
        # risk_counts = df['cvd_risk'].value_counts()

        # # Ordem desejada das categorias
        # risk_order = ['Low', 'Medium', 'High']

        # # Reindexar para garantir a ordem correta
        # risk_counts = risk_counts.reindex(risk_order)

        # # Definir cores na ordem certa
        # colors = ['green', 'orange', 'red']

        # plt.figure(figsize=(6,4))
        # bars = plt.bar(risk_counts.index, risk_counts.values, color=colors)

        # # Adicionar valores nas barras
        # for bar in bars:
        #     yval = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), ha='center')

        # plt.title("Data Distribution by CVD Risk")
        # plt.xlabel("Risk Level")
        # plt.ylabel("Number of Records")
        # plt.tight_layout()
        # plt.grid(axis='y', linestyle='--', alpha=0.6)
        # plt.show()
    
    case "ml_SVM":
        # Load data
        df = pd.read_csv("realistic_cvd_dataset2.csv")

        # Load selected features
        #with open("/home/eduardo/Desktop/master_project/Masters-Project/results_point_A__25_5/selected_features.json", "r") as f:
        with open("/home/eduardo/Desktop/master_project/Masters-Project/results_point_B__25_10/selected_features.json", "r") as f:
            selected_features = json.load(f)

        X_train = df[selected_features]
        y_train = df["cvd_risk_numeric"]

        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('svm', SVC()) # probability=True slows the ml by alot, more than 1h and still waiting
        ])

        param_dist = { # 1024 possible combinations
            'svm__C': [0.1, 0.2, 0.5, 1, 2, 5, 10, 100],
            'svm__kernel': ['linear', 'rbf'],
            'svm__gamma': ['scale', 'auto'],
            'svm__tol': [1e-1, 1e-2, 1e-3, 1e-4],
            'svm__shrinking': [True, False],
            'svm__class_weight': [None, 'balanced'],
            'svm__decision_function_shape': ['ovr'], # default value
            'svm__break_ties': [True, False]
        }

        tscv = TimeSeriesSplit(n_splits=10)

        scoring = {
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'balanced_accuracy': 'balanced_accuracy'
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_dist,
            scoring=scoring,
            refit='f1_weighted',
            cv=tscv,
            pre_dispatch=4,
            return_train_score=True
        )

        # Track training time
        start_time = time.time()
        grid_search.fit(np.ascontiguousarray(X_train, dtype=np.float64), y_train) # not much of a differenec in time execution. I will use it anyway to prevent copy
        execution_time = time.time() - start_time


        results = pd.DataFrame(grid_search.cv_results_)
        best_index = grid_search.best_index_

        best_results = results.loc[best_index, [
            'mean_test_f1_weighted', 'std_test_f1_weighted',
            'mean_test_precision_weighted', 'std_test_precision_weighted',
            'mean_test_recall_weighted', 'std_test_recall_weighted',
            'mean_test_balanced_accuracy', 'std_test_balanced_accuracy'
        ]]

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        print("Best estimator found:", grid_search.best_estimator_) # the full pipeline with best parameters
        print(f"Execution time: {execution_time/60:.2f} minutes")
        print(f"Best Scoring Metrics:{best_results}")
        print(f"Refit time using best model on whole dataset (sec):{grid_search.refit_time_}")

        # Create folder
        results_folder = 'svm_results_point_B_grid_search_10_contiguous'
        os.makedirs(results_folder, exist_ok=True)

        selected_features = X_train.columns.tolist()
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_
        best_cv_results = grid_search.cv_results_
        # Add execution time as a column in the full results
        best_cv_results['execution_time'] = execution_time/60


        # Save prints/results to a text file
        with open(os.path.join(results_folder, 'grid_search_results.txt'), 'w') as f:
            f.write(f"Selected features: {selected_features}\n")
            f.write(f"Best parameters found: {best_params}\n")
            f.write(f"Best estimator found: {best_estimator}\n")
            f.write(f"Best CV score found: {best_score}\n")
            f.write(f"Best Scoring Metrics:{best_results}")
            f.write(f"Best cv results found: {best_cv_results}\n")
            f.write(f"Refit time using best model on whole dataset (sec):{grid_search.refit_time_}")

    
        # Save the best model
        joblib.dump(grid_search.best_estimator_, os.path.join(results_folder, 'best_svm_pipeline.pkl'))

    case "ml_LR":
        # Load data
        df = pd.read_csv("realistic_cvd_dataset2.csv")

        # Load selected features
        #with open("/home/eduardo/Desktop/master_project/Masters-Project/results_point_A__25_5/selected_features.json", "r") as f:
        with open("/home/eduardo/Desktop/master_project/Masters-Project/results_point_B__25_10/selected_features.json", "r") as f:
            selected_features = json.load(f)

        X_train = df[selected_features]
        y_train = df["cvd_risk"]

        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('lr', LogisticRegression(multi_class='multinomial'))
        ])

        # Hyperparameter search space
        param_dist = { #960 possible combinations
            'lr__penalty': ['l2', None],
            'lr__tol': [1e-1, 1e-2, 1e-3, 1e-4], # Controls the stopping criterion. Smaller values = more precision but slower training.
            'lr__C': [0.3, 0.5, 1, 2, 3], # C param adjust the regularization (Low values: High regularization, allows some misclassifications, focusing on general margin. Better with noisy data. #High values: Low regularization, Tries to classify all points correctly, including outliers. Can lead to overfitting.)
            'lr__dual': [False], #Dual formulation is only implemented for l2 penalty with liblinear solver (only lbfgs and newton-cg available for this case). Prefer dual=False when n_samples > n_features.
            'lr__fit_intercept': [True, False],
            'lr__class_weight': [None, 'balanced'], # Class balancing (not impactful with balanced data)
            'lr__solver': ['lbfgs', 'newton-cg', 'sag'],
            'lr__warm_start': [True, False],
            'lr__max_iter': [100, 200, 300, 400]
        }

        scoring = {
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'balanced_accuracy': 'balanced_accuracy'
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=10)

        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_dist,
            scoring=scoring,
            refit='f1_weighted',
            cv=tscv,
            n_jobs=4
        )

        # Track training time
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        execution_time = time.time() - start_time

        results = pd.DataFrame(grid_search.cv_results_)
        best_index = grid_search.best_index_

        best_results = results.loc[best_index, [
            'mean_test_f1_weighted', 'std_test_f1_weighted',
            'mean_test_precision_weighted', 'std_test_precision_weighted',
            'mean_test_recall_weighted', 'std_test_recall_weighted',
            'mean_test_balanced_accuracy', 'std_test_balanced_accuracy'
        ]]

        print(f"Execution time: {execution_time / 60}")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        print("Best estimator found:", grid_search.best_estimator_) # the full pipeline with best parameters
        print(f"Best Scoring Metrics:{best_results}")


        # Create folder
        results_folder = 'lr_results_point_B_grid_search_10'
        os.makedirs(results_folder, exist_ok=True)

        selected_features = X_train.columns.tolist()
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_
        # Add execution time as a column in the full results
        best_cv_results = grid_search.cv_results_
        best_cv_results['execution_time_minutes'] = execution_time / 60


        # Save prints/results to a text file
        with open(os.path.join(results_folder, 'grid_search_results.txt'), 'w') as f:
            f.write(f"Selected features: {selected_features}\n")
            f.write(f"Best parameters found: {best_params}\n")
            f.write(f"Best estimator found: {best_estimator}\n")
            f.write(f"Best CV score found: {best_score}\n")
            f.write(f"Best Scoring Metrics:{best_results}")
            f.write(f"Best cv results found: {best_cv_results}\n")

        # Save best estimator
        joblib.dump(best_estimator, os.path.join(results_folder, 'best_lr_pipeline.pkl'))

    case "test":
        df = pd.read_csv("testing_dataset.csv")

        try:
            #best_model = joblib.load("/home/eduardo/Desktop/master_project/Masters-Project/Database_A/svm_results_point_A_grid_search_5/best_svm_pipeline.pkl")
            best_model = joblib.load("/home/eduardo/Desktop/master_project/Masters-Project/Database_B/svm_results_point_B_grid_search_7_contiguous/best_svm_pipeline.pkl")
            print("Loaded successfully!")
        except Exception as e:
            print("Error loading model:", e)

        # Load selected features
        #with open("/home/eduardo/Desktop/master_project/Masters-Project/Database_A/results_point_A__25_5/selected_features.json", "r") as f:
        with open("/home/eduardo/Desktop/master_project/Masters-Project/Database_B/results_point_B__25_10/selected_features.json", "r") as f:
            selected_features = json.load(f)

        X_test = df[selected_features]
        y_test = df['cvd_risk']

        y_pred = best_model.predict(X_test)

        # If predictions are strings, convert to numeric
        pred_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        if isinstance(y_pred[0], str):
            y_pred = pd.Series(y_pred).map(pred_mapping)

        # Ensure no NaNs in predictions
        mask = pd.notna(y_pred)
        X_test_clean = X_test[mask]
        y_test_clean = y_test[mask]
        y_pred_clean = y_pred[mask]


        # Compute balanced metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')        # weighted F1
        precision = precision_score(y_test, y_pred, average='weighted')  # weighted precision
        recall = recall_score(y_test, y_pred, average='weighted')        # weighted recall

        # Display results
        metrics = {
            "Balanced Accuracy": balanced_acc,
            "F1-score (weighted)": f1,
            "Precision (weighted)": precision,
            "Recall (weighted)": recall
        }

        print("Balanced Metrics on Test Set:")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")

        print(classification_report(y_test, y_pred))

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    case "T2D":
        # # Read the CSV
        # df = pd.read_csv("realistic_cvd_dataset1.csv")

        # # ✅ Filter only low-risk individuals
        # df_low = df[df['cvd_risk'] == 'High'].copy()

        # # ✅ Convert sleep time from seconds to hours
        # df_low['sleep_time_hr'] = df_low['sleep_time_sec'] / 3600

        # # ✅ Compute quartiles
        # q1_sec = df_low['sleep_time_sec'].quantile(0.25)
        # median_sec = df_low['sleep_time_sec'].median()
        # q3_sec = df_low['sleep_time_sec'].quantile(0.75)

        # q1_hr = df_low['sleep_time_hr'].quantile(0.25)
        # median_hr = df_low['sleep_time_hr'].median()
        # q3_hr = df_low['sleep_time_hr'].quantile(0.75)

        # # === Plot 1: Sleep time in seconds ===
        # fig1, ax1 = plt.subplots(figsize=(5, 4))
        # sns.boxplot(y=df_low['sleep_time_sec'], ax=ax1, color='lightblue')
        # ax1.set_title("Sleep Time (seconds)")
        # ax1.set_ylabel("Seconds")
        # ax1.set_xlabel("sleep_time")
        # ax1.grid(True)

        # # Annotate quartiles (Median bold, centered)
        # for y, label in zip([q1_sec, median_sec, q3_sec], ['Q1', 'Median', 'Q3']):
        #     font_weight = 'bold' if label == 'Median' else 'normal'
        #     ax1.text(0, y, f"{label}: {y:.1f}",
        #             ha='center', va='center', fontsize=9, color='black',
        #             fontweight=font_weight)

        # fig1.suptitle("Sleep Time (Hours)", fontsize=14)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()

        # # === Plot 2: Sleep time in hours ===
        # fig2, ax2 = plt.subplots(figsize=(5, 4))
        # sns.boxplot(y=df_low['sleep_time_hr'], ax=ax2, color='lightblue')
        # ax2.set_title("Sleep Time (Hours)")
        # ax2.set_ylabel("Hours")
        # ax2.set_xlabel("sleep_time")
        # ax2.grid(True)

        # # Annotate quartiles (Median bold, centered)
        # for y, label in zip([q1_hr, median_hr, q3_hr], ['Q1', 'Median', 'Q3']):
        #     font_weight = 'bold' if label == 'Median' else 'normal'
        #     ax2.text(0, y, f"{label}: {y:.2f}",
        #             ha='center', va='center', fontsize=9, color='black',
        #             fontweight=font_weight)

        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()
        print("t2d")

