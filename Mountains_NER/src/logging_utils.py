import os
import logging


def setup_logger():
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./logs")
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "training_metrics.log")

    logger = logging.getLogger("ner_eval")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file, mode='a')

        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()