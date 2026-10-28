# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
utils/logger.py

Centralized logging utility for molecular graph operations and debugging.

This module provides a reusable logging setup that writes to both console
and file, with consistent formatting. It ensures multiple modules and tools
can share a common logger instance without duplicating handlers.
"""

import logging
import os


class RelativePathFormatter(logging.Formatter):
    """
    Custom formatter that includes the relative file path in log messages.
    
    Converts absolute paths to relative paths from the project root,
    making logs more readable.
    """
    
    def __init__(self, fmt=None, datefmt=None, project_root=None):
        super().__init__(fmt, datefmt)
        # Use provided root or try to detect it
        if project_root:
            self.project_root = project_root
        else:
            # Default: go up from utils/ to project root
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def format(self, record):
        # Get the absolute path of the file that generated the log
        abs_path = record.pathname
        
        # Convert to relative path from project root
        try:
            rel_path = os.path.relpath(abs_path, self.project_root)
            # Normalize path separators for consistency
            rel_path = rel_path.replace("\\", "/")
        except ValueError:
            # On Windows, relpath fails if paths are on different drives
            rel_path = record.filename
        
        # Store original and set relative path
        record.relativepath = rel_path
        
        return super().format(record)


def get_logger(
    name: str = "chem",
    log_path: str = "run.log",
    level: int = logging.INFO,
    project_root: str = None
):
    """
    Creates and returns a logger with console and file handlers.
    
    Ensures that the logger is not re-initialized with duplicate handlers
    if called multiple times (singleton behavior by name).
    
    Args:
        name (str): Logger name (namespace). Defaults to 'chem'.
        log_path (str): Path to the output log file. Defaults to 'run.log'.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        project_root (str): Project root directory for relative paths.
                           If None, auto-detected from logger location.
    
    Returns:
        logging.Logger: Configured logger instance.
    
    Example:
        logger = get_logger(__name__)
        logger.info("Processing molecule")
        # Output: 2024-01-15 10:30:00 [INFO] [chem/build/create_molgraph.py:42] Processing molecule
    """
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(level)
        
        # Format includes relative file path and line number
        fmt = "%(asctime)s [%(levelname)s] [%(relativepath)s:%(lineno)d] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        
        formatter = RelativePathFormatter(fmt, datefmt, project_root)
        
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def get_logger_simple(
    name: str = "chem",
    log_path: str = "run.log",
    level: int = logging.INFO
):
    """
    Simpler logger using Python's built-in pathname formatting.
    
    Uses %(filename)s which gives just the file name (not full path).
    Lighter weight alternative if you don't need relative paths.
    """
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(level)
        
        fmt = "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
        formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")
        
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger