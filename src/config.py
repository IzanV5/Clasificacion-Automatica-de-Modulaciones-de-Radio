import os
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
MODELS_DIR = BASE_DIR / 'models' 
DATA_FILE = BASE_DIR / 'data' / 'dataset_completo.pkl'

# Etiquetas para la matriz de confusión
LABELS_ORDER = [
    'fax', 'lsb', 'usb', 'morse', 'mt63_1000',
    'dominoex11',
    'olivia16_500', 'olivia8_250', 'olivia32_1000', 'olivia16_1000', 
    'psk31', 'psk63', 'qpsk31', 
    'am', 'navtex', 'rtty100_850', 'rtty45_170', 'rtty50_170'
]