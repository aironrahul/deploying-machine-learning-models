import numpy as np
from sklearn.model_selection import train_test_split

#from regression_model.config import pipeline
#from regression_model.config import (load_dataset, save_pipeline)
from regression_model.config import config
#from regression_model.config import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = config.load_dataset(file_name=config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=0)  # we are setting the seed here

    # transform the target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    config.price_pipe.fit(X_train[config.FEATURES],
                            y_train)

    _logger.info(f'saving model version: {config.__version__}')
    config.save_pipeline(pipeline_to_persist=config.price_pipe)


#if __name__ == '__main__':
 #   run_training()
