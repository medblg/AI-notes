Intro machine learning - kaggle
===

1. how models work
---
- model ex -> decision tree -> basic building bloc
- capturing data -> fitting or training the model.
  - data -> training data
  
- improving decision tree:
  - deeper trees -> trees with more factors


2. basic data exploration
---
#### using pandas -> to get familiar with data
- 1st step any ml project -> familiarize yourself with data
- `pandas` -> exploring and manipulating data
  - `import pandas as pd`
  - `DataFrame` -> most important -> holds data as `table`
  - `data = pd.read_csv(melbourne_file_path)`
  - `data.describe()`

see pandas course :
https://www.kaggle.com/learn/pandas

3. 1st machine learning model
---
### selecting data for modeling
- prioritize variables ?
- see columns -> `columns` property of `DataFrame`
  - `data = pd.read_csv(melbourne_file_path)`
  - `data.columns`
- drops missing values -> `data2 = data.dropna(axis=0)` (na -> not available mnemo)

- selecting data : 2 ways:
  - dot notation -> `y = data.Price` -> used for prediction target
  - with column list -> -> for choosing features

- features : columns inputted into model(used to make predictions)
  - sometimes all columns except target. or fewer columns
  - by convention this data is called X
  - `data_features=['Rooms',Bathroom]`
  - `X = data[data_features]`
  - `X.describe()` and `X.head()`

### building model

- use `scikit-learn` written `sklearn` -> to build model
- steps:
  - define -> what model, decision tree.. ?
  - fit -> capture patterns from provided data.
  - predict 
  - evaluate -> determine how accurate the model's predictions are.
- ex:
```
from sklearn.tree import DecisionTreeRegressor

# specify same nbr in random_state -> ensures same results -> good practice
data_model = DecisionTreeRegressor(random_state=1)
data_model.fit(X, y) # will fit features(X) to target (y)

print(X.head())
print("predictions are: ")
print(data_model.predict(X.head()))
# data_model.predict(X.head())
```

4. model validation
---
- measures model's quality -> predictive accuracy
- model's prediction -> will be close ?
- mistakes -> predictions with training data & compare to target values in training data.

- looking through list of values -> pain -> need to summarize into single metrics.
- metrics for summarizing model quality:
  - Mean Absolute Error (MAE) -> positive nbr
    - ex: `error=actual−predicted`
```
from sklearn.metrics import mean_absolute_error

predicted_home_prices = data_model.predict(X)
# order doesn't matter
mean_absolute_error(y, predicted_home_prices)

```

- The prob with `In-Sample` scores :
  - used a single sample for building and evaluting the model
  - the model's job -> find patterns used to perform prediction

- measure performance on data that wasn't used to build model
  - exclude some data while building model. -> `validation data`
  - use `sklearn.model_selection` -> `train_test_split`
    - train data -> fit the model
    - validation data -> to calculate mean_absolute_error
```
from sklearn.model_selection import train_test_split

# splitting data into training and validation, for features and target
# random_state(10) -> to get same split every time using script
# train_X and train_y -> for training data -> to be fitted
# val_X and val_y -> for validation data -> val_X to be predicted
# MAE between validation target and predicted val_X
## order for MAE doesn't matter
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# define model
data_model = DecisionTreeRegressor(random_state=1)
data_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = data_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```
- improve model -> experimenting for better features || use diff model types

- is the MAE is good ? -> no general rule for this.

5. underfitting and overfitting
---
`https://www.kaggle.com/dansbecker/underfitting-and-overfitting`

- to make models more accurate

### experimenting with diff models

- decision tree -> more options -> 
`https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html`
  - important option `tree's depth` -> how many splits before coming to a prediciton

- `overfitting` -> model matches training data almost perfectly, but does poorly in validation and other new data.
- `underfitting` -> when a model fails to capture important distinctions and patterns in the data, so it perfoms poorly even in training data.

- `max_leaf_nodes` argument -> sensible way to control overfitting vs underfitting.

- we can use utility function to help compare MAE scores from diff values for `max_leaf_nodes`
```
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```
- we can use for-loop to compare accuracy of models with diff `max_leaf_nodes` values:
```
for max_leaf_nodes in [5, 50, 500, 5000]:
	my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
	print("max leaf nodes: %d \t\t Mean absolute error: %d" %(max_leaf_nodes, my_mae))
```

- **final model**: 
```
final_model = DecisionTreeRegressor(max_leaf_nodes = 100, random_state=0)
final_model.fit(X, y) ## with all the data
```

#### takeaway: 
- models suffer from either:
  - `overfitting` -> capturing patterns that won't recur in the future -> less accurate predic
  - `underfitting` -> failing to capture relevant patterns, -> less predic

- `validation data` -> not used in model training, it measures model's accuracy
  - lets try many candidate models -> keep the best

- Decision tree -> 
  - lot of leaves -> overfit. -> few data in its leaf
  - with few leaves -> shallow tree -> perform poorly -> fails to capture distinctions in raw data.
  
6. Random Forests 
---

- using sophisticated ML algorithms
- random forests -> many trees -> prediction by averaging prediction of each component tree.
- works well with default params

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

ex:
```
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
preds = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, preds)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
print("Validation MAE for Random Forest Model: %d" %(rf_val_mae))
```


- save predictions:
```
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```


Bonus lessons
---

8. intro to autoML
---
`https://www.kaggle.com/alexisbcook/intro-to-automl`

- 7 steps for ML
`https://towardsdatascience.com/the-7-steps-of-machine-learning-2877d7e5548e`
  1. gather data
  2. prepare data
  3. select model
  4. train model
  5. evaluate model
  6. tune parameters
  7. get predictions

===
===

Intermediate ML
===

- data types -> missing values, categorical variables
- design pipelines -> improve quality ml codes.
- use advanced tchnqs data validation -> cross-validation
- avoid common and impo data science mistakes (leakage)


#### Avoid missing values
- `scikit-learn` ->  errors when missing values.
- 3 strategies:
  1. drop columns with missing values
  2. imputation -> better option -> fills missing values with some nbre
    - ex: fills in with mean value
  3. an extension to imputation -> better predic considering which values were originally missing.

- function to measure quality of each approach -> calculate MAE:

```
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
```

- using imputation: `SimpleImputer`
```
from sklearn.impute import SimpleImputer
# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```
- approach 3 (extension to imputation):

```
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
```

- Ex:
```
# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)
```

#### categorical variables

- takes only limited nbr of values
- columns with text
```
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```
- 3 approaches:
  1. drop categ vrbles -> remove them from dataset, good if columns didn't contain useful information.
```
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```
  2. label encoding -> each unq value to diff integer
```
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
```
  3. one-hot-encoding -> create new columns indicating presence(absence) of each possible value in the orig data.
  
```
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```  
  

#### Pipelines

- to clean up modeling code
- bundles preprocessing and modeling steps -> as a single step
- benefits:
  - cleaner code
  - fewer bugs
  - easier to productionize
  - more options for model validation
  
- step:
`ColumnTransformer` -> to bundle diff preprocess steps

- step1: define preprocessing steps
```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```
- step2: define model
```
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```

- step3: create and evaluate Pipeline
```
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
```

#### cross-validation 

- for better measures of model's performance
- the larger the validation set -> less randomness (aka noise)
- cross-validation -> run our modeling process on diff subsets of the data to get multiple measures of model quality.
  - broke data for example into 5 **folds** -> data divided into 5 pieces
  - run 1 experiment for each fold
  - use every fold set as a `holdout set`  (validation set)

###### when use cross validation

- for small datasets
- for larger datasets -> single validation set is sufficient

ex:
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                               random_state=0))
                             ])
```

#### XGBoost
https://www.kaggle.com/alexisbcook/xgboost
- gradient boosting
- Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
