# src/utils/logger.py
import logging
import sys

# ANSI color codes for terminal output


class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


# Customize level names to be exactly 6 characters for alignment
logging.addLevelName(logging.DEBUG, "DEBUG ")    # 6 chars
logging.addLevelName(logging.INFO, "INFO  ")     # 6 chars
logging.addLevelName(logging.WARNING, "WARN  ")  # 6 chars
logging.addLevelName(logging.ERROR, "ERROR ")    # 6 chars
logging.addLevelName(logging.CRITICAL, "CRIT  ")  # 6 chars

# Add custom SUCCESS level
SUCCESS_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")  # 6 chars


def success(self, message, *args, **kwargs):
    """Log a message with severity 'SUCCESS'."""
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)


# Add the success method to Logger class
logging.Logger.success = success


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels"""

    LEVEL_COLORS = {
        'DEBUG ': Colors.BRIGHT_BLACK,      # Gray for debug
        'INFO  ': Colors.BRIGHT_BLUE,       # Blue for info
        'SUCCESS': Colors.BRIGHT_GREEN,      # Green for success
        'WARNING': Colors.BRIGHT_YELLOW,     # Yellow for warning
        'ERROR ': Colors.BRIGHT_RED,        # Red for error
        'CRITICAL': Colors.RED + Colors.BOLD,  # Bold red for critical
    }

    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record):
        # Get the standard formatted message
        message = super().format(record)

        if not self.use_colors or not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return message

        # Get the level name and apply color
        level_name = record.levelname
        color = self.LEVEL_COLORS.get(level_name, Colors.RESET)

        # Apply color to the level name in the message
        if level_name in message:
            colored_level = f"{color}{level_name}{Colors.RESET}"
            message = message.replace(level_name, colored_level, 1)

        return message


def setup_logger(
    name: str,
    level: int = logging.INFO,
    add_console_handler: bool = False,
    use_colors: bool = True
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
    use_colors : bool
        Whether to use colored output (default: True)

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
            formatter = ColoredFormatter(
                "[%(asctime)s] %(levelname)-7s - %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                use_colors=use_colors
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        else:
            # For library code, use NullHandler by default
            logger.addHandler(logging.NullHandler())

    return logger


def create_console_logger(name: str, level: int = logging.INFO, use_colors: bool = True) -> logging.Logger:
    """
    Create a logger with console output for end users.

    This is a convenience function for end users who want to see logs.

    Parameters
    ----------
    name : str
        Logger name
    level : int  
        Logging level
    use_colors : bool
        Whether to use colored output (default: True)
    """
    return setup_logger(name, level, add_console_handler=True, use_colors=use_colors)


def get_library_logger(name: str) -> logging.Logger:
    """
    Get a library logger (silent by default, controlled by end user).

    This is the recommended way for library modules to get loggers.
    """
    return setup_logger(name, add_console_handler=False)
