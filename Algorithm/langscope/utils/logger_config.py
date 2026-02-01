"""
Logging configuration for LangScope.

Provides consistent logging setup across all modules.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: str = None,
    log_file: str = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Log message format
        log_file: Optional file to write logs to
    
    Returns:
        Root logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger("langscope")
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (None for root langscope logger)
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"langscope.{name}")
    return logging.getLogger("langscope")


class LogContext:
    """
    Context manager for temporary log level changes.
    
    Example:
        with LogContext(level="DEBUG"):
            # Debug logging enabled here
            ...
    """
    
    def __init__(self, level: str = "DEBUG", logger_name: str = None):
        """
        Initialize log context.
        
        Args:
            level: Temporary log level
            logger_name: Logger to modify (None for root)
        """
        self.level = level
        self.logger_name = logger_name
        self.logger = get_logger(logger_name)
        self.original_level = None
    
    def __enter__(self):
        """Enter context - change log level."""
        self.original_level = self.logger.level
        numeric_level = getattr(logging, self.level.upper(), logging.DEBUG)
        self.logger.setLevel(numeric_level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore log level."""
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)
        return False


