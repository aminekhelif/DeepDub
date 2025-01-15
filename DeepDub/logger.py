# logger.py
import yaml
import logging
import os 
def setup_logger(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    logging_level = config.get('logging_level', 'INFO').upper()
    numeric_level = getattr(logging, logging_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {logging_level}')
    
    logging.basicConfig(level=numeric_level)
    logger = logging.getLogger(__name__)
    return logger

# actual loggeer path 
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
logger = setup_logger(config_path)
