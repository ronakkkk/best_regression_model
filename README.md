Pypi Library Link: https://pypi.org/project/bestregressionmodel/0.1.0/

Best Regression Model is used for supervised learning techniques where the target data is in continous form. It selects the best model from the eight regression model based on the Root Mean Square Value (RMSE). 

The eight regression model used in the given library are:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Random Forest Regression
6. Support Vector Regression
7. Extra Trees Regression
8. Decision Tree Regression

#### User installation

If you already have a working installation of numpy, scipy and sklearn, the easiest way to install best-classification-model is using pip

#### `pip install bestregressionmodel`

#### Important links

Official source code repo: https://github.com/ronakkkk/bestregressionmodel

Download releases: https://pypi.org/project/bestregressionmodel/

#### Examples
```import

from bestregressionmodel import regression_models

import pandas

data = pandas.read_csv('Data.csv')

X = data.iloc[:, :-1]

Y = data['Target']

best_model, best_model_name, acc = regression_models.reg_model(X, Y)

print(best_model_name, " (RMSE):", acc)```

`__Output__:

ElasticNet Regression (RMSE):621.2574962618987`

 
