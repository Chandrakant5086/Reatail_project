import pandas as pd
import joblib
import os
import sys
from sklearn.ensemble import RandomForestRegressor

# ensure project root on path when running script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger
from utils.exception_handler import handle_exception

logger = get_logger(__name__)

DATA_PATH = "data/processed/sales_clean.csv"
MODEL_PATH = "ml_model/sales_model.pkl"

try:
    logger.info("logged   cleaned  data for ML model training")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded data with shape {df.shape} and columns: {list(df.columns)}")

    x = df[['promotion', 'holiday']]
    y = df['sales']

    logger.info("initializing  the  random  forest  model")
    model=RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    logger.info("training the model")

    model.fit(x, y)

    # ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    logger.info('ML model trained and saved successfully')

except Exception as  e:
    logger.error ("ML training  failed", exc_info=True)
    raise  handle_exception(e)