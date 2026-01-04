import  logging
import  os 

LOG_DIR="logs"
os.makedirs(LOG_DIR,exist_ok=True)

# for the  configuration 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s |%(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR,"app.log")),
        logging.StreamHandler()
    ]
)

# create the object
logger = logging.getLogger("app_logger")

def get_logger(name: str):
    """Return a configured logger. Use get_logger(__name__) in modules."""
    return logging.getLogger(name)
