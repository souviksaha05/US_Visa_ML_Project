from dataclasses import dataclass
import os
import sys
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define the parameter grids for hyperparameter tuning
            param_grids = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "min_samples_split": [2, 5, 10]
                },
                "Logistic Regression": {
                    "penalty": ["l2", "none"],
                    "C": [0.1, 1, 10],
                    "solver": ["liblinear", "saga"]
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                },
                "XGBClassifier": {
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1, 0.2]
                },
                "CatBoosting Classifier": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "iterations": [50, 100, 200],
                    "l2_leaf_reg": [1, 3, 5],
                    "border_count": [32, 50, 100]
                },
                "Support Vector Classifier": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"],
                    "degree": [3, 4, 5]
                },
                "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                }
            }
            
            # Define the models
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(), 
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "Support Vector Classifier": SVC(),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            best_models = {}
            for name, model in models.items():
                logging.info(f"Running RandomizedSearchCV for {name}...")
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grids[name],
                    n_iter=100,
                    cv=3,
                    verbose=2,
                    n_jobs=-1,
                    scoring='accuracy'
                )
                random_search.fit(X_train, y_train)
                best_models[name] = random_search.best_estimator_

                logging.info(f"Best params for {name}: {random_search.best_params_}")

            # Evaluate tuned models
            tuned_report = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=best_models,
                param_grids=param_grids  
            )

            # Select the best model based on the report
            best_model_name = tuned_report.sort_values(by='Accuracy', ascending=False).iloc[0]['Model Name']
            best_model = best_models[best_model_name]

            # Check if the best model meets the minimum accuracy threshold
            if tuned_report[tuned_report['Model Name'] == best_model_name]['Accuracy'].values[0] < 0.6:
                raise CustomException("No model meets the minimum accuracy requirement")

            logging.info(f"Best found model: {best_model_name} with accuracy: {tuned_report[tuned_report['Model Name'] == best_model_name]['Accuracy'].values[0]}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model

        except Exception as e:
            raise CustomException(e, sys)
