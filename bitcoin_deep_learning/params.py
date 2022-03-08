import os
from dotenv import load_dotenv
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
################################################################################
#               cross-validation params
################################################################################
FOLD_TRAIN_SIZE = 12*30
FOLD_TEST_SIZE= 3*30
HORIZON  = 7
load_dotenv()
API_KEY = os.getenv('API_KEY')
# gap = HORIZON-1
# sequence_lenght = 90
# fold_step = 30
# sample_step = 1
################################################################################
#
################################################################################
