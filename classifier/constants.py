from utils.constants import TRAIN, VALIDATION
# Analysis metrics and labels.
# ----------------------------
TRUES = 'trues'
PREDS = 'preds'
PRECISION = 'precision'
RECALL = 'recall'
F1_SCORE = 'f1_score'
CONFUSION = 'confusion_matrix'
EPOCH_TIME = 'epoch_time'
LOSS = 'loss'
LOSS_DIFF = 'loss_diff'

# Hyperparameters
# ---------------
# Optimizer parameters
LR = 'lr'
BETAS = 'betas'

# Model parameters
FILTERS = 'filters'
POOL_SIZE = 'pool_size'
FULLY_CONNECTED = 'fully_connected'
DROPOUT = 'drop'

# Hyperparameter search dataframe schema
# --------------------------------------

SEARCH_COLUMN_SCHEMA = [
    # Hyperparameters
    LR,
    BETAS,
    FILTERS,
    POOL_SIZE,
    FULLY_CONNECTED,
    DROPOUT,

    # Evaluation metrics
    TRAIN + '_' + LOSS,
    VALIDATION + '_' + LOSS,
    LOSS_DIFF,
    TRAIN + '_' + PRECISION,
    VALIDATION + '_' + PRECISION,
    TRAIN + '_' + RECALL,
    VALIDATION + '_' + RECALL,
    TRAIN + '_' + F1_SCORE,
    VALIDATION + '_' + F1_SCORE,
    EPOCH_TIME,
]
