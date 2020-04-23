import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from pprint import pprint
from category_encoders import OneHotEncoder


def get_best_split(data, data_encoded, target_col, categorical_cols, regression_algorithm):
    best_split = None
    for feature in data.columns:
        feature_values, counts = np.unique(data[feature], return_counts=True)

        # CATEGORICAL COLUMN #
        if feature in categorical_cols:
            data_splits, data_encoded_splits, regression_models, MSEs = [], [], [], []
            for feature_value in feature_values:
                data_split = data.loc[data[feature] == feature_value]
                data_splits.append(data_split.reset_index(drop=True))
                data_encoded_split = data_encoded.loc[data_split.index]
                data_encoded_splits.append(data_encoded_split.reset_index(drop=True))
                regression_model, MSE = fit_regression_model(data_encoded_split, target_col, regression_algorithm)
                regression_models.append(regression_model)
                MSEs.append(MSE)
            weighted_MSE = np.dot(MSEs, counts)
            if best_split is None or weighted_MSE < best_split[-1]:  # if weighted_MSE is new best
                best_split = [feature, feature_values, data_splits, data_encoded_splits, regression_models, MSEs,
                              weighted_MSE]
        # NUMERICAL COLUMN #
        else:
            for feature_value in feature_values:
                data_split_left = data.loc[data[feature] < feature_value]
                data_split_right = data.loc[data[feature] >= feature_value]
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
                weighted_MSE = np.dot(MSEs, [len(data_split_left), len(data_split_right)])
                if best_split is None or weighted_MSE < best_split[-1]:  # if weighted_MSE is new best
                    best_split = [feature, feature_values, data_splits, data_encoded_splits, regression_models, MSEs,
                                  weighted_MSE]

    return best_split


def fit_regression_model(data, target_col, regression_algorithm):
    x = data.drop(columns=[target_col])
    y = data[target_col]
    regression_model = regression_algorithm()
    regression_model.fit(x, y)
    y_pred = regression_model.predict(x)
    return regression_model, mean_squared_error(y, y_pred)


def regression_tree(data, data_encoded, target_col, categorical_cols, min_instances_to_split, regression_algorithm,
                    parent_regression_model=None, curr_regression_model=None, curr_MSE=None):

    if len(data) < min_instances_to_split or len(np.unique(data[target_col])) == 1:  # stopping criteria
        return parent_regression_model
    if curr_regression_model is None:  # true only in tree's root
        curr_regression_model, curr_MSE = fit_regression_model(data_encoded, target_col, regression_algorithm)

    # get best split
    best_split = get_best_split(data, data_encoded, target_col, categorical_cols, regression_algorithm)
    best_feature, split_names, data_splits, data_encoded_splits, regression_models, MSEs, weighted_MSE = best_split
    if weighted_MSE > curr_MSE:  # if best split is worse than current node
        return curr_regression_model

    # perform split
    node = {best_feature: {}}
    for i in range(len(data_splits)):
        child_node = regression_tree(data_splits[i], data_encoded_splits[i], target_col, categorical_cols,
                                     min_instances_to_split, regression_algorithm, curr_regression_model,
                                     regression_models[i], MSEs[i])
        node[best_feature][split_names[i]] = child_node  # add the child node to tree
    return node


def predict(query, tree, default=1):
    # 1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            # 2.
            try:
                result = tree[key][query[key]]
            except:
                return default

            # 3.
            result = tree[key][query[key]]
            # 4.
            if isinstance(result, dict):
                return predict(query, result)

            else:
                return result


def train_test_split(dataset, categorical_cols, train_fraction):
    dataset_encoded = OneHotEncoder(cols=categorical_cols, use_cat_names=True).fit_transform(dataset)
    train_len = len(dataset) * train_fraction
    train_set = dataset.sample(n=train_len, random_state=1)
    train_set_encoded = dataset_encoded.loc[train_set.index].reset_index(drop=True)
    test_set = dataset.drop(train_set.index).reset_index(drop=True)
    test_set_encoded = dataset_encoded.drop(train_set.index).reset_index(drop=True)
    return train_set.reset_index(drop=True), train_set_encoded, test_set, test_set_encoded


def test(tree, data, data_encoded):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = data.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data["class"]) / len(data)) * 100, '%')


# dataset settings
dataset = pd.read_csv("datasets/bike_sharing.csv", usecols=['season', 'holiday', 'weekday', 'weathersit', 'cnt'])
target_col = 'cnt'
categorical_cols = []

# model parameters
min_instances_to_split = 5
regression_algorithm = linear_model.LinearRegression

# train and test model
train_fraction = 0.9
train_set, train_set_encoded, test_set, test_set_encoded = train_test_split(dataset, categorical_cols, train_fraction)
tree = regression_tree(train_set, train_set_encoded, target_col, categorical_cols, min_instances_to_split,
                       regression_algorithm)
pprint(tree)
test(tree, test_set, test_set_encoded)
