import  joblib 
import  pandas  as  pd 
from utils.logger import  get_logger

logger=get_logger(__name__)

MODEL_PATH="ml_model/sales_model.pkl"

def  predict_sales(promotion:int, holiday:int):
    logger.info ("loading trained model  for  inference")
    model=joblib.load(MODEL_PATH)

    input_df=pd.DataFrame([{
        "promotion":promotion,
        "holiday":holiday
    }])

    prediction=model.predict(input_df)[0]
    logger.info(f"predication generated :{prediction}")
    return prediction

if __name__=="__main__":
    result=predict_sales (promotion=1,holiday=0)
    print (f"predicted sales:{result}")