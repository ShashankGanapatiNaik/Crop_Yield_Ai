import logging
import sys

from .config import get_settings


def setup_logging() -> None:
    logging.basicConfig(level=get_settings().log_level, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", force=True)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # we log requests ourselves
