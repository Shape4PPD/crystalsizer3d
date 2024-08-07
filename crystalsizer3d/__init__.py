import logging
import os
import sys
import time
from pathlib import Path

import dotenv

# Timestamp for when the script was started, can be used for log names
START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')

# Get running environment
ENV = os.getenv('ENV', 'local')

# Set base path to point to the repository root
ROOT_PATH = Path(__file__).parent.parent

# Load environment variables from .env file
dotenv.load_dotenv(ROOT_PATH / '.env')

# Set script path
cwd = Path.cwd()
dir_name = os.path.dirname(sys.argv[0]).replace(str(cwd), '').lstrip('/\\')
SCRIPT_PATH = (cwd / dir_name).resolve()
if not SCRIPT_PATH.is_relative_to(ROOT_PATH):
    ROOT_PATH = SCRIPT_PATH
    dotenv.load_dotenv(ROOT_PATH / '.env')


def _load_env_path(k: str, default: Path):
    ep = os.getenv(k)
    if ep is None:
        ep = default
    if ep is not None:
        ep = Path(ep)
    return ep


# Data directory
DATA_PATH = _load_env_path('DATA_PATH', ROOT_PATH / 'data')

# Number of parallel workers to use for tasks
N_WORKERS = int(os.getenv('N_WORKERS', 8))

# PyTorch to use JIT where possible
PYTORCH_JIT = os.getenv('PYTORCH_JIT', 1)

# Use CUDA?
USE_CUDA = bool(os.getenv('USE_CUDA', '1').lower() in ['1', 'true', 'yes', 'y'])

# Check for mitsuba default variants
MI_CPU_VARIANT = os.getenv('MI_CPU_VARIANT', 'llvm_ad_rgb')

# Dataset proxy
DATASET_PROXY_PATH = _load_env_path('DATASET_PROXY_PATH', DATA_PATH / 'dataset_proxies')

# CSD proxy
CSD_PROXY_PATH = _load_env_path('CSD_PROXY_PATH', DATA_PATH / 'csd_proxy.json')

# Use mlab?
USE_MLAB = bool(os.getenv('USE_MLAB', '1').lower() in ['1', 'true', 'yes', 'y'])

# || -------------------------------- LOGS --------------------------------- ||

LOGS_PATH = ROOT_PATH / 'logs' / SCRIPT_PATH.relative_to(ROOT_PATH) / Path(sys.argv[0]).stem
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
WRITE_LOG_FILES = os.getenv('WRITE_LOG_FILES', False)

# Set formatting
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

# Create a logger with the name corresponding to the script being executed
script_name = os.path.basename(sys.argv[0])[:-3]
logger = logging.getLogger(script_name)
logger.setLevel(LOG_LEVEL)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Setup handlers
if WRITE_LOG_FILES:
    LOG_FILENAME = f'{script_name}_{time.strftime("%Y-%m-%d_%H%M%S")}.log'
    print(f'Writing logs to: {LOGS_PATH}/{LOG_FILENAME}')
    os.makedirs(LOGS_PATH, exist_ok=True)
    file_handler = logging.FileHandler(f'{LOGS_PATH}/{LOG_FILENAME}', mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Don't propagate logs to the root logger as this causes duplicate entries
logger.propagate = False


# Handle uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.critical(
        'Uncaught exception',
        exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = handle_exception

# || ---------------------------------------------------------------------- ||
