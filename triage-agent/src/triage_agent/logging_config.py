import logging
import os


def configure_logging() -> None:
    log_level_name = os.environ.get(
        "LOG_LEVEL",
        "INFO",
    ).upper()

    log_level = getattr(
        logging,
        log_level_name,
        logging.INFO,
    )

    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
    )

    for logger_name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)