import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# linear regression model
def linear_regression(feature_train, feature_test, target_train, target_test):
    start = time.clock()

    # train the model
    linreg = LinearRegression()
    linreg.fit(feature_train, target_train)
    print(linreg.coef_, linreg.intercept_)

    # test the model
    target_prediction = linreg.predict(feature_test)

    # evaluate the model
    evaluate_model(target_test, target_prediction)

    # calculate runtime
    elapsed = (time.clock() - start)
    print('Time used:', elapsed)

    # draw the graph
    draw_graph(target_test, target_prediction, 'linear regression')


# neural network model
def neural_network(feature_train, feature_test, target_train, target_test):
    start = time.clock()

    # standard scale preprocess
    scaler = StandardScaler()
    scaler.fit(feature_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    feature_train = scaler.transform(feature_train)
    feature_test = scaler.transform(feature_test)

    # train the model
    mlp = MLPRegressor(hidden_layer_sizes=(1000,), alpha=0.1, solver='lbfgs', max_iter=500)
    mlp.fit(feature_train, target_train)

    # test the model
    target_prediction = mlp.predict(feature_test)

    # evaluate the model
    evaluate_model(target_test, target_prediction)

    # calculate runtime
    elapsed = (time.clock() - start)
    print('Time used:', elapsed)

    # draw the graph
    draw_graph(target_test, target_prediction, 'Neural Network')


def neural_network_param_test(feature_train, target_train):
    param_test = {'hidden_layer_size': range(100, 1100, 100)}
    gsearch = GridSearchCV(MLPRegressor(alpha=0.1, solver='lbfgs', max_iter=500), param_grid=param_test, scoring='r2', iid=False, cv=None)
    gsearch.fit(feature_train, target_train)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)


# decision tree model
def decision_tree(feature_train, feature_test, target_train, target_test):
    start = time.clock()

    # train the model
    dec_tree = DecisionTreeRegressor(max_depth=14)
    dec_tree.fit(feature_train, target_train)

    # test the model
    target_prediction = dec_tree.predict(feature_test)

    # evaluate the model
    evaluate_model(target_test, target_prediction)

    # calculate runtime
    elapsed = (time.clock() - start)
    print('Time used:', elapsed)

    # draw the graph
    draw_graph(target_test, target_prediction, 'Decision Tree')


# adjust best parameters for decision tree
def decision_tree_param_test(feature_train, target_train):
    param_test = {'max_depth': range(5, 51, 1)}
    gsearch = GridSearchCV(DecisionTreeRegressor(), param_grid=param_test, scoring='r2', iid=False, cv=None)
    gsearch.fit(feature_train, target_train)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)


# gradient boosting regression tree model
def gradient_boosting_regression_tree(feature_train, feature_test, target_train, target_test):
    start = time.clock()

    # train the model
    gra_boo_reg_tree = GradientBoostingRegressor(n_estimators=1200, learning_rate=1, max_depth=4)
    gra_boo_reg_tree.fit(feature_train, target_train)

    # test the model
    target_prediction = gra_boo_reg_tree.predict(feature_test)

    # evaluate the model
    evaluate_model(target_test, target_prediction)

    # calculate runtime
    elapsed = (time.clock() - start)
    print('Time used:', elapsed)

    # draw the graph
    draw_graph(target_test, target_prediction, 'Gradient Boosting Regression Tree')


# adjust best parameters for gradient boosting regression tree
def gradient_boosting_regression_tree_param_n_estimators_test(feature_train, target_train):
    param_test = {'n_estimators': range(100, 3100, 100)}
    gsearch = GridSearchCV(GradientBoostingRegressor(learning_rate=1), param_grid=param_test,
                           scoring='r2', iid=False, cv=None)
    gsearch.fit(feature_train, target_train)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)


def gradient_boosting_regression_tree_param_max_depth_test(feature_train, target_train):
    param_test = {'max_depth': range(2, 9, 1)}
    gsearch = GridSearchCV(GradientBoostingRegressor(n_estimators=1200, learning_rate=1), param_grid=param_test,
                           scoring='r2', iid=False, cv=None)
    gsearch.fit(feature_train, target_train)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)


# random forest model
def random_forest(feature_train, feature_test, target_train, target_test):
    start = time.clock()

    # train the model
    rand_tree = RandomForestRegressor(oob_score=True, n_estimators=90, max_depth=24)
    rand_tree.fit(feature_train, target_train)

    # test the model
    target_prediction = rand_tree.predict(feature_test)

    # evaluate the model
    evaluate_model(target_test, target_prediction)

    # calculate runtime
    elapsed = (time.clock() - start)
    print('Time used:', elapsed)

    # draw the graph
    draw_graph(target_test, target_prediction, 'random forest')


# adjust best parameters for random forest
def random_forest_param_n_estimators_test(feature_train, target_train):
    param_test = {'n_estimators': range(10, 110, 10)}
    gsearch = GridSearchCV(RandomForestRegressor(oob_score=True), param_grid=param_test,
                           scoring='r2', iid=False, cv=None)
    gsearch.fit(feature_train, target_train)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)


def random_forest_param_max_depth_test(feature_train, target_train):
    param_test = {'max_depth': range(11, 31, 1)}
    gsearch = GridSearchCV(RandomForestRegressor(n_estimators=90, oob_score=True), param_grid=param_test,
                           scoring='r2', iid=False, cv=None)
    gsearch.fit(feature_train, target_train)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)


def evaluate_model(target_test, target_prediction):
    print('MSE', mean_squared_error(target_test, target_prediction))
    print('RMSE', np.sqrt(mean_squared_error(target_test, target_prediction)))
    print('R2', r2_score(target_test, target_prediction))


def draw_graph(target_test, target_prediction, graph_name):
    fig, ax = plt.subplots()
    ax.scatter(target_test, target_prediction)
    ax.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Measured \n ' + graph_name)
    plt.show()


def main():
    # get data
    data = pd.read_csv('./demand_prediction_data.csv')
    data = data[pd.notnull(data["count"])]
    data["workingday"] = data.weekday.map({1: 1,
                                               2: 1,
                                               3: 1,
                                               4: 1,
                                               5: 1,
                                               6: 0,
                                               0: 0})

    categoryVariableList = ["hour","day","month","icon","weekday","holiday","workingday"]
    for var in categoryVariableList:
        data[var] = data[var].astype("category")

    feature = data[["hour","day","month","icon","holiday","workingday","holiday","workingday","temperature", "humidity", "pressure", "windSpeed", "windBearing", "visibility",
           "restaurant", "bus", "subway"]]

    target = data[["count"]]
    # feature = data[['month', 'dow', 'weekday', 'holiday', 'temperature_max', 'temperature_min', 'precipitation',
          # 'average_wind_speed', 'snow_fall', 'snow_depth', 'capacity', 'nyct_gid', 'taxi_zone_id',
          # 'subway_num', 'bus_stop_num']]
    # target = data[['count']]

    # spilt train dataset and test dataset
    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.7, random_state=1)

    # build model
    # linear_regression(feature_train, feature_test, target_train, target_test)
    neural_network(feature_train, feature_test, target_train, target_test)
    # decision_tree(feature_train, feature_test, target_train, target_test)
    # gradient_boosting_regression_tree(feature_train, feature_test, target_train, target_test)
    # random_forest(feature_train, feature_test, target_train, target_test)


if __name__ == "__main__":
    main()
