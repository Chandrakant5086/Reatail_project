import traceback
from utils.logger import logger

def handle_exception(error: Exception) -> Exception:
    # 'error' is the exception object
    logger.error("Exception occurred")
    logger.error(f"Error Message: {error}")
    # traceback.format_exc() returns the full stack trace as a string
    logger.error(traceback.format_exc())

    # return an Exception instance so callers can 'raise handle_exception(e)'
    return Exception(f"An error occurred: {error}")