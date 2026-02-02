import logging
import os
import sys
import warnings

warnings.filterwarnings(action="ignore", message="divide by zero encountered in divide", lineno=759)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.getLevelName(os.getenv("LOGLEVEL", "INFO").upper()),
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
