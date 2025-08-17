import logging, logging.config, os


def configure_logging(level: str = "INFO", json_format: bool = True):
    if json_format:
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "json",
                }
            },
            "loggers": {"": {"handlers": ["stdout"], "level": level}},
        }
    else:
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "std": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
            },
            "handlers": {
                "stdout": {"class": "logging.StreamHandler", "formatter": "std"}
            },
            "loggers": {"": {"handlers": ["stdout"], "level": level}},
        }
    logging.config.dictConfig(config)
