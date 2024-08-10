import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder, PowerTransformer,LabelEncoder



from src.exception import CustomException
from src.logger import logging
import os
from imblearn.combine import SMOTETomek, SMOTEENN
from src.utils import save_object
from datetime import date

CURRENT_YEAR = date.today().year

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
       
        '''
        This function is responsible for data transformation
        
        '''
        try:
            
            numerical_columns = ['no_of_employees', 'company_age']
            categorical_columns = [ 'case_id', 'continent', 'education_of_employee', 
            'has_job_experience', 'requires_job_training', 
            'region_of_employment', 'unit_of_wage', 'full_time_position', 'case_status']

            or_columns = ['has_job_experience', 'requires_job_training', 'full_time_position', 'education_of_employee']
            oh_columns = ['continent', 'unit_of_wage', 'region_of_employment']

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, ['company_age']),
                    ("StandardScaler", numeric_transformer, numerical_columns)
                ]
            )


            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "case_status"
            numerical_columns = ['no_of_employees', 'yr_of_estab', 'prevailing_wage']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
            input_feature_train_df.drop('yr_of_estab', inplace=True, axis=1)

            logging.info(f"input_feature_train_df columns: {input_feature_train_df.columns}")
            logging.info(f"input_feature_train_df head: \n{input_feature_train_df.head()}")

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
            input_feature_test_df.drop('yr_of_estab', inplace=True, axis=1)

            logging.info(f"input_feature_test_df columns: {input_feature_test_df.columns}")
            logging.info(f"input_feature_test_df head: \n{input_feature_test_df.head()}")# Apply Label Encoding to target variable
           
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            logging.info("Label Encoding applied on target variable")

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")


            
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            # Fit and transform the training data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Transform the testing data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Data transformation completed successfully.")

            # Apply SMOTEENN
            smt = SMOTEENN(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
          raise CustomException(e, sys)

        