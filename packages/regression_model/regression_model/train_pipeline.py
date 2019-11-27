import numpy as np
from sklearn.model_selection import train_test_split

from regression_model import pipeline
from regression_model.processing.data_management import (
    load_dataset, save_pipeline)
from regression_model.config import config
from regression_model import __version__ as _version

import logging


from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.processing import preprocessors as pp
#from regression_model.processing import features
#from regression_model.config import config

#import logging


_logger = logging.getLogger(__name__)


price_pipe = Pipeline(
    [
        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),
        ('numerical_inputer',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),
        ('temporal_variable',
            pp.TemporalVariableEstimator(
                variables=config.TEMPORAL_VARS,
                reference_variable=config.DROP_FEATURES)),
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.01,
                variables=config.CATEGORICAL_VARS)),
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('log_transformer',
            pp.LogTransformer(variables=config.NUMERICALS_LOG_VARS)),
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
        ('scaler', MinMaxScaler()),
        ('Linear_model', Lasso(alpha=0.005, random_state=0))
    ]
)





_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=0)  # we are setting the seed here

    # transform the target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    pipeline.price_pipe.fit(X_train[config.FEATURES],
                            y_train)

    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)


if __name__ == '__main__':
    run_training()
