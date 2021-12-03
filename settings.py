from os.path import join, abspath, dirname

BASE_DIR = dirname(abspath(__file__))

ASSET_DIR = join(BASE_DIR, 'assets')

MODEL_SAVE_DIR = join(BASE_DIR, 'models', 'saved_models')
MODEL_RESULT_DIR = join(BASE_DIR, 'models', 'results')
MODEL_TENSORBOARD_DIR = join(BASE_DIR, 'models', 'tensorboard_log')
