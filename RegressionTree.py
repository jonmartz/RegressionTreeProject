import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from pprint import pprint
from category_encoders import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def regression_tree(data, data_encoded, target_col, categorical_cols, min_samples_split, regression_algorithm,
                    parent_regression_model=None, curr_regression_model=None, curr_MSE=None):
    if len(data.index) < min_samples_split or len(np.unique(data[target_col])) == 1:  # stopping criteria
        return parent_regression_model
    if curr_regression_model is None:  # true only in tree's root
        curr_regression_model, curr_MSE = fit_regression_model(data_encoded, target_col, regression_algorithm)

    # get best split
    best_split = get_best_split(data, data_encoded, target_col, categorical_cols, regression_algorithm)
    feature, feature_values, data_splits, data_encoded_splits, regression_models, MSEs, weighted_MSE = best_split
    if weighted_MSE > curr_MSE:  # if best split is worse than current node
        return curr_regression_model

    # perform split
    child_nodes = {}
    node = {'feature_name': feature, 'child_nodes': child_nodes, 'regression_model': curr_regression_model}
    for i in range(len(data_splits)):
        child_node = regression_tree(data_splits[i], data_encoded_splits[i], target_col, categorical_cols,
                                     min_samples_split, regression_algorithm, curr_regression_model,
                                     regression_models[i], MSEs[i])
        child_nodes[feature_values[i]] = child_node  # add the child node to tree
    return node


def get_best_split(data, data_encoded, target_col, categorical_cols, regression_algorithm):
    best_split = None
    for feature in data.columns.drop(target_col):
        feature_values, counts = np.unique(data[feature], return_counts=True)

        # CATEGORICAL COLUMN #
        if feature in categorical_cols:
            data_splits, data_encoded_splits, regression_models, MSEs = [], [], [], []
            split_names = feature_values
            for feature_value in feature_values:
                data_split = data.loc[data[feature] == feature_value]
                data_splits.append(data_split.reset_index(drop=True))
                data_encoded_split = data_encoded.loc[data_split.index]
                data_encoded_splits.append(data_encoded_split.reset_index(drop=True))
                regression_model, MSE = fit_regression_model(data_encoded_split, target_col, regression_algorithm)
                regression_models.append(regression_model)
                MSEs.append(MSE)
            # weighted_MSE = np.dot(MSEs, counts)
            weighted_MSE = np.average(MSEs, weights=counts)
            if best_split is None or weighted_MSE < best_split[-1]:  # if weighted_MSE is new best
                best_split = [feature, split_names, data_splits, data_encoded_splits, regression_models, MSEs,
                              weighted_MSE]
        # NUMERICAL COLUMN #
        else:
            for feature_value in feature_values:
                data_split_left = data.loc[data[feature] < feature_value]
                data_split_right = data.loc[data[feature] >= feature_value]
                if 0 in [len(data_split_left.index), len(data_split_right.index)]:
                    continue  # if no splitting was actually performed
                data_splits = [data_split_left.reset_index(drop=True), data_split_right.reset_index(drop=True)]
                data_encoded_split_left = data_encoded.loc[data_split_left.index].reset_index(drop=True)
                data_encoded_split_right = data_encoded.loc[data_split_right.index].reset_index(drop=True)
                data_encoded_splits = [data_encoded_split_left, data_encoded_split_right]
                regression_model_left, MSE_left = fit_regression_model(data_encoded_split_left, target_col,
                                                                       regression_algorithm)
                regression_model_right, MSE_right = fit_regression_model(data_encoded_split_right, target_col,
                                                                         regression_algorithm)
                regression_models = [regression_model_left, regression_model_right]
                MSEs = [MSE_left, MSE_right]
                weighted_MSE = np.average(MSEs, weights=[len(data_split_left.index), len(data_split_right.index)])
                if best_split is None or weighted_MSE < best_split[-1]:  # if weighted_MSE is new best
                    split_names = ['<%s' % feature_value, '>=%s' % feature_value]
                    best_split = [feature, split_names, data_splits, data_encoded_splits, regression_models, MSEs,
                                  weighted_MSE]
    return best_split


def fit_regression_model(data, target_col, regression_algorithm):
    x = data.drop(columns=[target_col])
    y = data[target_col]
    regression_model = regression_algorithm()
    regression_model.fit(x, y)
    y_pred = regression_model.predict(x)
    return regression_model, mean_squared_error(y, y_pred)


def predict(node, data, data_encoded, target_col, categorical_cols):
    y, y_pred = [], []
    if len(data.index) == 0:
        return y, y_pred
    feature = node['feature_name']
    child_nodes = node['child_nodes']
    feature_values = child_nodes.keys()
    if feature in categorical_cols:  # in case the test set has categories that are absent in the train set
        unknown_categories_data = data[~data[feature].isin(feature_values)]
        if len(unknown_categories_data) > 0:
            y.extend(unknown_categories_data[target_col].tolist())
            unknown_categories_data_encoded = data_encoded.loc[unknown_categories_data.index]
            y_pred.extend(node['regression_model'].predict(unknown_categories_data_encoded).tolist())
    for feature_value in feature_values:
        if feature in categorical_cols:
            data_split = data.loc[data[feature] == feature_value]
        else:  # feature is numerical
            if feature_value[0] == '<':
                data_split = data.loc[data[feature] < float(feature_value[1:])]
            else:  # starts with >=
                data_split = data.loc[data[feature] >= float(feature_value[2:])]
        if len(data_split.index) > 0:
            data_encoded_split = data_encoded.loc[data_split.index]
            child_node = child_nodes[feature_value]
            if type(child_node) is dict:  # continue down the tree
                child_y, child_y_pred = predict(child_node, data_split, data_encoded_split, target_col, categorical_cols)
            else:  # child_node is a leaf, i.e., a regression model
                child_y = data_split[target_col].tolist()
                child_y_pred = child_node.predict(data_encoded_split).tolist()
            y.extend(child_y)
            y_pred.extend(child_y_pred)
    return y, y_pred


def train_test_split(dataset, categorical_cols, train_fraction):
    dataset_encoded = OneHotEncoder(cols=categorical_cols, use_cat_names=True).fit_transform(dataset)
    train_len = int(len(dataset.index) * train_fraction)
    train_set = dataset.sample(n=train_len, random_state=1)
    train_set_encoded = dataset_encoded.loc[train_set.index].reset_index(drop=True)
    test_set = dataset.drop(train_set.index).reset_index(drop=True)
    test_set_encoded = dataset_encoded.drop(train_set.index).reset_index(drop=True)
    return train_set.reset_index(drop=True), train_set_encoded, test_set, test_set_encoded


def test(tree_root, data, data_encoded, target_col, categorical_cols):
    simple_regression_MSE = fit_regression_model(data_encoded, target_col, regression_algorithm)[1]
    print('simple logistic regression MSE = %.5f' % simple_regression_MSE)
    y, y_pred = predict(tree_root, data, data_encoded.drop(columns=target_col), target_col, categorical_cols)
    return mean_squared_error(y, y_pred)


def sklearn_test(train_set, test_set, target_col, min_samples_split):
    x_train = train_set.drop(columns=target_col)
    y_train = train_set[target_col]
    x_test = test_set.drop(columns=target_col)
    y_test = test_set[target_col]
    tree = DecisionTreeRegressor(random_state=1)
    # if i == 0:
    #     tree = DecisionTreeRegressor(random_state=1)
    # elif i == 1:
    #     tree = DecisionTreeRegressor(min_samples_split=min_samples_split, random_state=1)
    # else:
    #     tree = DecisionTreeRegressor(min_samples_split=min_samples_split, min_samples_leaf=min_samples_split, random_state=1)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    return mean_squared_error(y_test, y_pred)


# todo: choose dataset settings and model parameters
dataset = pd.read_csv("datasets/bike_sharing.csv", usecols=['season', 'holiday', 'weekday', 'weathersit', 'cnt'])
target_col = 'cnt'
categorical_cols = []
train_fraction = 0.9
# MODEL PARAMETERS REQUESTED IN ASSIGNMENT:
min_samples_split = 30
regression_algorithm = linear_model.LinearRegression

# train and test model
train_set, train_set_encoded, test_set, test_set_encoded = train_test_split(dataset, categorical_cols, train_fraction)
print('building tree...')
tree_root = regression_tree(train_set, train_set_encoded, target_col, categorical_cols, min_samples_split,
                            regression_algorithm)
print('testing tree...')
our_MSE = test(tree_root, test_set, test_set_encoded, target_col, categorical_cols)
print('our MSE = %.5f' % our_MSE)
sklearn_MSE = sklearn_test(train_set_encoded, test_set_encoded, target_col, min_samples_split)
print('sklearn MSE = %.5f' % sklearn_MSE)
print('our MSE is %.4f%% larger than sklearn MSE' % ((our_MSE/sklearn_MSE - 1) * 100))
