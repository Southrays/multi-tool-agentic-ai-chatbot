
import logging


def setup_logging():
    """
    Configure application-wide logging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the requested module.
    """
    return logging.getLogger(name)
