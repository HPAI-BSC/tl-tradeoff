from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


# PATH-RELATED CONSTANTS
BASE_PATH = '/gpfs/projects/bsc70/hpai/storage/data/atormos/'
MODEL_PATH = '/gpfs/projects/bsc70/hpai/storage/data/brownie/models'
CROPS_PATH = '/gpfs/projects/bsc70/hpai/storage/data/datasets/256px/'


# EXECUTION-RELATED CONSTANTS
NUM_CROPS = 10
ALL_DATASETS = [
    'caltech101',
    'cub200',
    'textures',
    'food101',
    'mame',
    'mit67',
    'wood',
    'flowers102',
    'catsdogs',
    'stanforddogs'
]
SAMPLES_PER_DATASET = {
    'Caltech 101': [1, 2, 3, 5, 7, 10, 15, 20],
    'Oxford-IIIT-Pet': [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 83],
    'CUB200': [1, 2, 3, 5, 7, 10, 15, 20],
    'Oxford Flower': [1, 2, 3, 5, 7, 10],
    'Food-101': [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100, 150, 190],
    'MAMe': [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200],
    'MIT ISR': [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 67],
    'Stanford Dogs': [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 90],
    'DTD': [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70],
    'Knots6': [1, 2, 3, 5, 7, 10, 15, 18],
}
ACTUAL_DATASET_NAMES = {
    'caltech101': 'Caltech 101',
    'cub200': 'CUB200',
    'textures': 'DTD',
    'food101': 'Food-101',
    'mame': 'MAMe',
    'mit67': 'MIT ISR',
    'wood': 'Knots6',
    'flowers102': 'Oxford Flower',
    'catsdogs': 'Oxford-IIIT-Pet',
    'stanforddogs': 'Stanford Dogs'
}
CLASSES_PER_DATASET = {
    'Oxford Flower': 102,
    'Oxford-IIIT-Pet': 37,
    'MIT ISR': 67,
    'CUB200': 200,
    'DTD': 47,
    'Food-101': 101,
    'MAMe': 29,
    'Stanford Dogs': 120,
    'Caltech 101': 101,
    'Knots6': 6
}
CLASSIFIERS = {
    'LinearSVC': LinearSVC(),
    'SVC': SVC(),
    'SGDClassifier': SGDClassifier(tol=5e-4),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier()
}
FE_MODEL_SELECTION_SPLITS_PER_DATASET = {
    'Oxford Flower': 1,
    'Oxford-IIIT-Pet': 1,
    'MIT ISR': 6,
    'CUB200': 4,
    'DTD': 1,
    'Food-101': 19,
    'MAMe': 8,
    'Stanford Dogs': 10,
    'Caltech 101': 1,
    'Knots6': 1
}