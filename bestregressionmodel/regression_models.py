from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

def reg_model(X, Y):
    '''
    This function helps to define different regression models and return the best regression model based on the accuracy score.
    :param X: Feature
    :param Y: Target
    :return: Model, Model_Name and Accuracy Score
    '''
    # split data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y)

    # Dictionary to store accuracy score of each model
    reg_acc = {}

    '''Linear Regression'''
    linear_model = LinearRegression()

    # Fit model
    linear_model.fit(Xtrain, ytrain)

    # predict
    ypred = linear_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['Linear Regression'] = mean_squared_error(ypred, ytest, squared=False)

    '''Ridge Regression'''
    ridge_model = Ridge()

    # Fit model
    ridge_model.fit(Xtrain, ytrain)

    # predict
    ypred = ridge_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['Ridge Regression'] = mean_squared_error(ypred, ytest, squared=False)

    '''Lasso Regression'''
    lasso_model = Lasso()

    # Fit model
    lasso_model.fit(Xtrain, ytrain)

    # predict
    ypred = lasso_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['Lasso Regression'] = mean_squared_error(ypred, ytest, squared=False)

    '''ElasticNet Regression'''
    elasticnet_model = ElasticNet()

    # Fit model
    elasticnet_model.fit(Xtrain, ytrain)

    # predict
    ypred = elasticnet_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['ElasticNet Regression'] = mean_squared_error(ypred, ytest, squared=False)

    '''Support Vector Regression Regression'''
    svr_model = SVR()

    # Fit model
    svr_model.fit(Xtrain, ytrain)

    # predict
    ypred = svr_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['SVR Regression'] = mean_squared_error(ypred, ytest, squared=False)

    '''Poisson Regression'''
    poisson_regressor_model = PoissonRegressor()

    # Fit model
    poisson_regressor_model.fit(Xtrain, ytrain)

    # predict
    ypred = poisson_regressor_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['Poisson Regression'] = mean_squared_error(ypred, ytest, squared=False)

    '''Random Forest Regression'''
    rfr_model = RandomForestRegressor()

    # Fit model
    rfr_model.fit(Xtrain, ytrain)

    # predict
    ypred = rfr_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['Random Forest Regression'] = mean_squared_error(ypred, ytest, squared=False)

    '''Extra Trees Regression'''
    extra_tree_model = ExtraTreesRegressor()

    # Fit model
    extra_tree_model.fit(Xtrain, ytrain)

    # predict
    ypred = extra_tree_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['Extra Trees Regression'] = mean_squared_error(ypred, ytest, squared=False)

    '''Decision Tree Regression'''
    dec_tree_model = DecisionTreeRegressor()

    # Fit model
    dec_tree_model.fit(Xtrain, ytrain)

    # predict
    ypred = dec_tree_model.predict(Xtest)

    # using rmse to deduce best model
    reg_acc['Decision Tree Regression'] = mean_squared_error(ypred, ytest, squared=False)

    # Finding key with maximum accuracy value
    best_model = min(reg_acc, key=reg_acc.get)

    models = {"Linear Regression": linear_model, "Ridge Regression": ridge_model, "Lasso Regression": lasso_model, "ElasticNet Regression": elasticnet_model,
              "SVR Regression": svr_model, "Poisson Regression": poisson_regressor_model, "Random Forest Regression": rfr_model,
              "Extra Trees Regression": extra_tree_model, "Decision Tree Regression": dec_tree_model}
    
    return models[best_model], best_model,reg_acc[best_model]
