import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from category_encoders import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def regression_tree(data, data_encoded, target_col, categorical_cols, min_samples_split, regression_algorithm,
                    parent_regression_model=None, curr_regression_model=None, curr_MSE=None):
    """
    Our implementation of the regression tree. For numeric columns performs the best binary split and for categorical
    columns splits by every possible category. Saves a regression model in each node in case that the test set
    contains an instance with a category that wasn't present in the train set.
    :param data: to be trained on, includes the target class.
    :param data_encoded: one-hot-encoded of data
    :param target_col: col to be predicted
    :param categorical_cols: list of the column names of the categorical columns (previously identified automatically)
    :param min_samples_split: minimum number of samples in node to perform a split
    :param regression_algorithm: to put in every node
    :param parent_regression_model: regression model from parent node, in case the created node has too few samples
    :param curr_regression_model: regression model of current node sent by the parent who already computed it for MSE
    :param curr_MSE: MSE obtained by curr_regression_model, sent by the parent who already computed it
    :return: current node (first function call returns the tree's root)
    """
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
    """
    Finds the best possible split from current node.
    :param data: to be trained on, includes the target class.
    :param data_encoded: one-hot-encoded of data
    :param target_col: col to be predicted
    :param categorical_cols: list of the column names of the categorical columns (previously identified automatically)
    :param regression_algorithm: to put in every node
    :return: a list containing:
            [
            feature = name of column selected for split,
            split_names = string describing the feature value in each split path,
            data_splits = data to be sent to each split path,
            data_encoded_splits = one-hot-encoded version of data_splits,
            regression_models = regression models to be sent to each split path,
            MSEs = MSEs obtained from regression_models,
            weighted_MSE = weighted average of the MSEs from each split path
            ]
    """
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
    """
    Fits a regression model the implements regression_algorithm
    :param data: one-hot-encoded of data to be trained on
    :param target_col: col to be predicted
    :param regression_algorithm: algorithm to implement for regression
    :return: a list [fitted regression model, MSE obtained]
    """
    x = data.drop(columns=[target_col])
    y = data[target_col]
    regression_model = regression_algorithm()
    regression_model.fit(x, y)
    y_pred = regression_model.predict(x)
    return regression_model, mean_squared_error(y, y_pred)


def predict(node, data, data_encoded, target_col, categorical_cols):
    """
    Recursive implementation of querying a tree for predictions. To speed up the process, only one recursive call is
    made for every split path relevant to the data, where all the data relevant to each split path is sent to the
    child function call.
    :param node: root of regression tree
    :param data: to obtain predictions for, includes target class column
    :param data_encoded: one-hot-encoded of data
    :param target_col: col to be predicted
    :param categorical_cols: list of the column names of the categorical columns (previously identified automatically)
    :return: a list [target column, predicted target column], may not be in the same order as in original data
    """
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
    """
    Splits the dataset into a train and a test set
    :param dataset: data to be split
    :param categorical_cols: list of the column names of the categorical columns (previously identified automatically)
    :param train_fraction: portion of dataset to be used as train set
    :return: a list [train set, one-hot-encoded train set, test set, one-hot-encoded test set]
    """
    dataset_encoded = OneHotEncoder(cols=categorical_cols, use_cat_names=True).fit_transform(dataset)
    train_len = int(len(dataset.index) * train_fraction)
    train_set = dataset.sample(n=train_len, random_state=1)
    train_set_encoded = dataset_encoded.loc[train_set.index].reset_index(drop=True)
    test_set = dataset.drop(train_set.index).reset_index(drop=True)
    test_set_encoded = dataset_encoded.drop(train_set.index).reset_index(drop=True)
    return train_set.reset_index(drop=True), train_set_encoded, test_set, test_set_encoded


def test(tree_root, data, data_encoded, target_col, categorical_cols):
    """
    Test the performance of a regression tree. All this does is perform the first recursive call of the
    predict function.
    :param tree_root: root of regression tree
    :param data: to obtain predictions for, includes target class column
    :param data_encoded: one-hot-encoded of data
    :param target_col: col to be predicted
    :param categorical_cols: list of the column names of the categorical columns (previously identified automatically)
    :return: the MSE of the tree's predictions
    """
    y, y_pred = predict(tree_root, data, data_encoded.drop(columns=target_col), target_col, categorical_cols)
    return mean_squared_error(y, y_pred)


def simple_and_sklearn_test(train_set, test_set, target_col, min_samples_split, regression_algorithm):
    """
    For comparing our tree's performance with a simple regression model, AKA a single-node (the root) regression tree,
    and with sklearn's regression tree implementation that employs a regular linear regression model.
    :param train_set: one-hot-encoded version of train set
    :param test_set: one-hot-encoded version of test set
    :param target_col: target class columns name
    :param min_samples_split: minimum number of samples in node to perform a split
    :param regression_algorithm: to be implemented by regression model
    :return: a list [MSE of simple regression model, MSE of sklearn's tree]
    """
    x_test = test_set.drop(columns=target_col)
    y_test = test_set[target_col]

    # train and test a simple regression model, AKA single node regression tree
    simple_model = fit_regression_model(train_set_encoded, target_col, regression_algorithm)[0]
    y_pred = simple_model.predict(x_test)
    simple_MSE = mean_squared_error(y_test, y_pred)

    # train and test sklearn's implementation of regression trees
    sklearn_tree = DecisionTreeRegressor(random_state=1, min_samples_split=min_samples_split)
    sklearn_tree.fit(train_set.drop(columns=target_col), train_set[target_col])
    y_pred = sklearn_tree.predict(x_test)
    sklearn_MSE = mean_squared_error(y_test, y_pred)

    return simple_MSE, sklearn_MSE


# todo: choose dataset settings and model parameters here
dataset = pd.read_csv("datasets/zoo.csv")
target_col = 'class'
train_fraction = 0.9
# MODEL PARAMETERS REQUESTED IN ASSIGNMENT:
min_samples_split = 30
regression_algorithm = linear_model.LinearRegression
# regression_algorithm = linear_model.SGDRegressor
# regression_algorithm = linear_model.Ridge

# train and test model
categorical_cols = dataset.columns[[not np.issubdtype(i, np.number) for i in dataset.dtypes]].to_list()
train_set, train_set_encoded, test_set, test_set_encoded = train_test_split(dataset, categorical_cols, train_fraction)
print('building tree...')
tree_root = regression_tree(train_set, train_set_encoded, target_col, categorical_cols, min_samples_split,
                            regression_algorithm)
our_MSE = test(tree_root, test_set, test_set_encoded, target_col, categorical_cols)
simple_MSE, sklearn_MSE = simple_and_sklearn_test(train_set_encoded, test_set_encoded, target_col, min_samples_split,
                                                  regression_algorithm)
print('simple regression MSE = %.5f' % simple_MSE)
print('sklearn MSE = %.5f' % sklearn_MSE)
print('our MSE = %.5f' % our_MSE)
print('our MSE is %.4f%% larger than sklearn MSE' % ((our_MSE/sklearn_MSE - 1) * 100))
