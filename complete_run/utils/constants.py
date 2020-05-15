# Mean signal and sequence length cutoffs.
MEAN_SIGNAL = 0.5
SEQUENCE_LENGTH = 100
TOTAL_CLASSES = 16

# Name of sequence column added to full dataframe.
RAW_SEQUENCE = 'raw_sequence'

# DHS data column names.
SEQNAME = 'seqname'
START = 'start'
END = 'end'
DHS_WIDTH = 'DHS_width'
SUMMIT = 'summit'
TOTAL_SIGNAL = 'total_signal'
NUMSAMPLES = 'numsamples'
NUMPEAKS = 'numpeaks'

DHS_DATA_COLUMNS = [
    SEQNAME,
    START,
    END,
    DHS_WIDTH,
    SUMMIT,
    TOTAL_SIGNAL,
    NUMSAMPLES,
    NUMPEAKS,
]

# NMF component related columns.
COMPONENT = 'component'
PROPORTION = 'proportion'
COMPONENT_COLUMNS = [
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
    'C6',
    'C7',
    'C8',
    'C9',
    'C10',
    'C11',
    'C12',
    'C13',
    'C14',
    'C15',
    'C16',
]

# Holdout chromosomes used for Test and Validation datasets.
TEST_CHR = 'chr1'
VALIDATION_CHR = 'chr2'

REFERENCE_GENOME_FILETYPE = 'fasta'

# Data npy file naming constants.
# Labels:
TRAIN = 'train'
TEST = 'test'
VALIDATION = 'validation'
# Kinds:
SEQUENCES = 'sequences'
COMPONENTS = 'components'
# Models:
GENERATOR = 'generator'
CLASSIFIER = 'classifier'

# Train, test and validation dataset constants.
NUM_SEQS_PER_COMPONENT = {
    TRAIN: 10000,
    TEST: 1000,
    VALIDATION: 1000,
}

def data_filename(label, kind, model):
    return f"{label}_{kind}_{model}.npy"
