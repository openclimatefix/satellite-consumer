import warnings
import os
import logging
import sys

warnings.filterwarnings(action="ignore", message="divide by zero encountered in divide", lineno=759)
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(os.getenv("LOGLEVEL", "INFO").upper()))
