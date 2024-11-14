import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

           # Creating numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))  # Disable centering for sparse matrices
    ]
)

          # Creating categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=True)),  # Ensure sparse output
                    ("scaler", StandardScaler(with_mean=False))  # Disable centering for sparse matrices
    ]
)


            # Logging the columns for reference
            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            # Combining the transformations using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            # Defining the target and feature columns
            target_column_name = "math_score" 
            numerical_columns = ["writing_score", "reading_score"]

            # Splitting the data into features and target variable
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")
            try:
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                logging.info("Transformation of training data successful.")
            except Exception as e:
                logging.error(f"Error during training data transformation: {e}")
                raise

            try:
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                logging.info("Transformation of test data successful.")
            except Exception as e:
                logging.error(f"Error during test data transformation: {e}")
                raise



            # Combining the transformed features with the target variable
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object to disk...")
            save_object(
                        file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                    )
            logging.info(f"Preprocessing object saved at {self.data_transformation_config.preprocessor_obj_file_path}")


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
