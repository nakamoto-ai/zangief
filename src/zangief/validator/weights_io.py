from loguru import logger
import os 
import json

def ensure_weights_file(zangief_dir_name, weights_file_name):
    if not os.path.exists(zangief_dir_name):
        os.makedirs(zangief_dir_name)
        logger.info(f"Created directory: {zangief_dir_name}")

    if not os.path.exists(weights_file_name):
        with open(weights_file_name, 'w') as file:
            json.dump({}, file)
        logger.info(f"Created file: {weights_file_name}")

