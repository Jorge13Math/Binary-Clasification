import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')

logger = logging.getLogger(__name__)


def genererate_dataframe(path):
    """
    Generate dataframe from folder dataset

    Args:
        :path: Folder's Path

    Returns:
        :df_train: Dataframe with values to train a model
        :df_test: Dataframe with  values to test the model

    """
    logger.info('Loading files')
    train_file = 'accounts_train.csv'
    quotes_train_file = 'quotes_train.csv'
    test_file = 'accounts_test.csv'
    quotes_test_file = 'quotes_test.csv'

    train = pd.read_csv(path + train_file)
    quotes_train = pd.read_csv(path + quotes_train_file)

    test = pd.read_csv(path + test_file)
    quotes_test = pd.read_csv(path + quotes_test_file)

    df_train = pd.merge(quotes_train, train, on='account_uuid', how="left")
    df_test = pd.merge(quotes_test, test, on='account_uuid', how="left")
    logger.info('Dataframes generated')

    return df_train, df_test
