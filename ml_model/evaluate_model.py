import pandas as pd
import joblib
from  sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy  as np
from utils.logger import get_logger

logger=get_logger(__name__)

DATA_PATH="data/processed/sales_clean.csv"
MODEL_PATH="ml_model/sales_model.pkl"

df=pd.read_csv(DATA_PATH)
model=joblib.load(MODEL_PATH)


x=df[['promotion','holiday']]
y_true=df['sales']

y_pred=model.predict(x)

MAE=mean_absolute_error(y_true,y_pred)
RMSE=np.sqrt(mean_squared_error(y_true,y_pred))
r2=r2_score(y_true,y_pred)

logger.info(f"MAE:{MAE}")
logger.info(f"RMSE:{RMSE}")
logger.info(f"r2 score:{r2}")

print(f"MAE:{MAE}")
print(f"RMSE:{RMSE}")
print(f"r2 score:{r2}")