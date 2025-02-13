"""Setup logging configuration for the application."""

import json
import sys
import os
import loguru

def development_formatter(record: "loguru.Record") -> str:
    """Format a log record for development."""
    return "".join((
        "<green>{time:HH:mm:ss.SSS}</green> ",
        "<level>{level:<7s}</level> [{file}:{line}] | <level>{message}</level> ",
        "<green>{extra}</green>" if record["extra"] else "",
        "\n{exception}",
    ))

def structured_formatter(record: "loguru.Record") -> str:
    """Format a log record as a structured JSON object."""
    # TODO: Make exceptions dict-formatted when present
    record["extra"]["serialized"] = json.dumps({
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "severity": record["level"].name,
        "message": record["message"],
        "logging.googleapis.com/labels": {"python_logger": record["name"]},
        "logging.googleapis.com/sourceLocation": {
            "file": record["file"].name,
            "line": record["line"],
            "function": record["function"],
        },
        "context": record["extra"],
    })
    return "{extra[serialized]}\n"

# Define the logging formatter, removing the default one
loguru.logger.remove(0)
if sys.stdout.isatty():
    # Simple logging for development
    loguru.logger.add(
        sys.stdout, format=development_formatter, diagnose=True,
        level=os.getenv("LOGLEVEL", "DEBUG"), backtrace=True, colorize=True,
    )
else:
    # JSON logging for containers
    loguru.logger.add(
        sys.stdout, format=structured_formatter, backtrace=True,
        level=os.getenv("LOGLEVEL", "INFO").upper(),
    )

# Uncomment and change the list to quieten external libraries
# for logger in ["aiobotocore", "cfgrib"]:
#    logging.getLogger(logger).setLevel(logging.WARNING)

