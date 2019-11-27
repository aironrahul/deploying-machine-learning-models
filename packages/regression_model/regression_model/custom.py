#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################config###########################
import os
import pathlib

#import regression_model

#import pandas as pd
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import numpy as np
#import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
#import numpy as np
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
#import numpy as np
from sklearn.model_selection import train_test_split



pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


#PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
PACKAGE_ROOT = pathlib.Path.cwd()
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'test.csv'
TRAINING_DATA_FILE = 'train.csv'
TARGET = 'SalePrice'


# variables
FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood',
            'OverallQual', 'OverallCond', 'YearRemodAdd',
            'RoofStyle', 'MasVnrType', 'BsmtQual', 'BsmtExposure',
            'HeatingQC', 'CentralAir', '1stFlrSF', 'GrLivArea',
            'BsmtFullBath', 'KitchenQual', 'Fireplaces', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive',
            'LotFrontage',
            # this one is only to calculate temporal variable:
            'YrSold']

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = 'YrSold'

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['MasVnrType', 'BsmtQual', 'BsmtExposure',
                            'FireplaceQu', 'GarageType', 'GarageFinish']

TEMPORAL_VARS = 'YearRemodAdd'

# variables to log transform
NUMERICALS_LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea']

# categorical variables to encode
CATEGORICAL_VARS = ['MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType',
                    'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
                    'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'PavedDrive']

# NUMERICAL_NA_NOT_ALLOWED = [
#     feature for feature in FEATURES
#     if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
# ]

# CATEGORICAL_NA_NOT_ALLOWED = [
#     feature for feature in CATEGORICAL_VARS
#     if feature not in CATEGORICAL_VARS_WITH_NA
# ]


PIPELINE_NAME = 'lasso_regression'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'

# used for differential testing
#ACCEPTABLE_MODEL_DIFFERENCE = 0.05


############data management#################

# import pandas as pd
# from sklearn.externals import joblib
# from sklearn.pipeline import Pipeline

#from regression_model.config import config
#from regression_model import __version__ as _version

#import logging


#_logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str ) :
    #_data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    _data = pd.read_csv(f'{DATASET_DIR}/{file_name}')
    return _data


def save_pipeline(*, pipeline_to_persist) :

    # Prepare versioned save file name
    #save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_file_name = f'{PIPELINE_SAVE_FILE}.pkl'
    #save_path = config.TRAINED_MODEL_DIR / save_file_name
    save_path = TRAINED_MODEL_DIR / save_file_name

    #remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    #_logger.info(f'saved pipeline: {save_file_name}')


def load_pipeline(*, file_name: str) :
    

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep) :

    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, '__init__.py']:
            model_file.unlink()

            

###############Preprocessors######################


# import numpy as np
# import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin

#from regression_model.processing.errors import InvalidModelInputError


class CategoricalImputer(BaseEstimator, TransformerMixin):
    

    def __init__(self, variables=None) :
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) :
        
        return self

    def transform(self, X: pd.DataFrame) :
        

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    

    def __init__(self, variables=None, reference_variable=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variables = reference_variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variables] - X[feature]

        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.encoder_dict_[feature]), X[feature], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(
                ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if transformer introduces NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {key: value for (key, value) in null_counts.items()
                     if value is True}
            raise InvalidModelInputError(
                f'Categorical encoder has introduced NaN when '
                f'transforming categorical variables: {vars_.keys()}')

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X

    
##############Features################
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin

#from regression_model.processing.errors import InvalidModelInputError


class LogTransformer(BaseEstimator, TransformerMixin):
    

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        # check that the values are non-negative for log transform
        if not (X[self.variables] > 0).all().all():
            vars_ = self.variables[(X[self.variables] <= 0).any()]
            raise InvalidModelInputError(
                f"Variables contain zero or negative values, "
                f"can't apply log for vars: {vars_}")

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X





# In[ ]:


#############pipeline##############

# from sklearn.linear_model import Lasso
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler

#from regression_model.processing import preprocessors as pp
#from regression_model.processing import features
#from regression_model.config import config

#import logging


#_logger = logging.getLogger(__name__)


price_pipe = Pipeline(
    [
        ('categorical_imputer',
            CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)),
        ('numerical_inputer',
            NumericalImputer(variables=NUMERICAL_VARS_WITH_NA)),
        ('temporal_variable',
            TemporalVariableEstimator(
                variables=TEMPORAL_VARS,
                reference_variable=DROP_FEATURES)),
        ('rare_label_encoder',
            RareLabelCategoricalEncoder(
                tol=0.01,
                variables=CATEGORICAL_VARS)),
        ('categorical_encoder',
            CategoricalEncoder(variables=CATEGORICAL_VARS)),
        ('log_transformer',
            LogTransformer(variables=NUMERICALS_LOG_VARS)),
        ('drop_features',
            DropUnecessaryFeatures(variables_to_drop=DROP_FEATURES)),
        ('scaler', MinMaxScaler()),
        ('Linear_model', Lasso(alpha=0.005, random_state=0))
    ]
)


##################train pipeline##########################
# import numpy as np
# from sklearn.model_selection import train_test_split

# from regression_model import pipeline
# from regression_model.processing.data_management import (
#     load_dataset, save_pipeline)
# from regression_model.config import config
# from regression_model import __version__ as _version

#import logging


#_logger = logging.getLogger(__name__)


def run_training() -> None:
    

    # read training data
    data = load_dataset(file_name=TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.1,
        random_state=0)  # we are setting the seed here

    # transform the target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    price_pipe.fit(X_train[FEATURES],y_train)

    #_logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=price_pipe)


# if __name__ == '__main__':
#     run_training()


# In[ ]:


##############Predict#############################

# import numpy as np
# import pandas as pd

# from regression_model.processing.data_management import load_pipeline
# from regression_model.config import config
# from regression_model.processing.validation import validate_inputs
# from regression_model import __version__ as _version

# import logging


# _logger = logging.getLogger(__name__)

#pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
pipeline_file_name = f'{PIPELINE_SAVE_FILE}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) :
    

    data = pd.read_json(input_data)
    #validated_data = validate_inputs(input_data=data)
    #prediction = _price_pipe.predict(validated_data[FEATURES])
    prediction = _price_pipe.predict(data[FEATURES])
    output = np.exp(prediction)

    results = {'predictions': output, 'version': _version}

#     _logger.info(
#         f'Making predictions with model version: {_version} '
#         f'Inputs: {validated_data} '
#         f'Predictions: {results}')

    return results

if __name__ == '__main__':
    run_training()

