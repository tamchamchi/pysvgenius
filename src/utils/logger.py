# src/utils/logger.py
import logging
import sys


def setup_logger(
    name: str,
    level: int = logging.INFO,
    add_console_handler: bool = False
) -> logging.Logger:
    """
    Setup a module-specific logger with standard format.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    add_console_handler : bool
        Whether to add console handler. For libraries, usually False
        to let end users control logging output.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # Avoid duplicate handlers
        logger.setLevel(level)

        if add_console_handler:
            # Only add console handler if explicitly requested
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        else:
            # For library code, use NullHandler by default
            logger.addHandler(logging.NullHandler())

    return logger


def create_console_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with console output for end users.

    This is a convenience function for end users who want to see logs.
    """
    return setup_logger(name, level, add_console_handler=True)


def get_library_logger(name: str) -> logging.Logger:
    """
    Get a library logger (silent by default, controlled by end user).

    This is the recommended way for library modules to get loggers.
    """
    return setup_logger(name, add_console_handler=False)
