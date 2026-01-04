import pandas as pd
import os
import sys

# Ensure project root is on Python path when running this script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger
from utils.exception_handler import handle_exception

logger = get_logger(__name__)

RAW_PATH = "data/raw/sales_data.csv"
PROCESSED_PATH = "data/processed/sales_clean.csv"

try:
    logger.info("Starting the data cleaning process")

    # load the data
    df = pd.read_csv(RAW_PATH)
    logger.info(f"Raw data shape: {df.shape}")

    # remove duplicates
    df = df.drop_duplicates()

    # handle missing values
    df = df.dropna()

    # convert to standard data types
    df['date'] = pd.to_datetime(df['date'])
    df['promotion'] = df['promotion'].astype(int)
    df['holiday'] = df['holiday'].astype(int)
    df['sales'] = df['sales'].astype(float)

    logger.info('Data types standardized')

    # business rule: keep positive sales only
    df = df[df['sales'] > 0]

    # select machine learning required columns
    final_df = df[["promotion", "holiday", "sales"]]

    # ensure processed directory exists
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    final_df.to_csv(PROCESSED_PATH, index=False)

    logger.info('Cleaned data saved for the machine learning model')

except Exception as e:
    logger.error('Data cleaning failed')
    raise handle_exception(e)