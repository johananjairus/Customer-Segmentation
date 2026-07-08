import os
import logging
import sys

def setup_logging(log_file: str = "output/segmentation.log") -> logging.Logger:
    """
    Sets up custom logging that outputs to both a file and the console.
    
    Parameters:
    -----------
    log_file : str
        The file path where log records should be saved.
        
    Returns:
    --------
    logging.Logger
        Configured logger instance.
    """
    # Create the directories for the log file if they do not exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Define log formats
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Root logger configuration
    logger = logging.getLogger("CustomerSegmentation")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file handler for logging. Error: {e}", file=sys.stderr)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info("Logging system initialized.")
    return logger


def ensure_directories(base_path: str = ".") -> None:
    """
    Ensures that all standard project directories exist.
    
    Parameters:
    -----------
    base_path : str
        The base path of the project. Default is current directory '.'.
    """
    directories = [
        os.path.join(base_path, "data"),
        os.path.join(base_path, "notebooks"),
        os.path.join(base_path, "src"),
        os.path.join(base_path, "output")
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created folder: {directory}")
