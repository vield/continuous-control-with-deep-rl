import logging
import sys


def initialize_logging(stdout_logging_level='DEBUG', log_to_stdout=True):
    """Initializes a root logger and a stdout log handler.

    :param stdout_logging_level:
        One of the allowed logging levels.
    :param log_to_stdout:
        Whether to initializer the stdout log handler at all.
    """
    assert stdout_logging_level in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

    if log_to_stdout:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        stdout_logging_level = getattr(logging, stdout_logging_level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(stdout_logging_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d %b %Y %H:%M:%S"
        )
        handler.setFormatter(formatter)

        root_logger.addHandler(handler)