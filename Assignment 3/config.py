# ====   PATHS ===================

TRAINING_DATA_FILE = "titanic.csv"
PIPELINE_NAME = 'logistic_regression.pkl'


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

NUMERICAL_VARS_TO_IMPUTE = ['age', 'fare']

CABIN = 'cabin'
