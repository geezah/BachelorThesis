import pprint
from time import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from utils import *

pp = pprint.PrettyPrinter()

RAW_PATH = '../../datasets/raw.csv'
CF_PATH = "../../datasets/crafted_features.csv"


def load_csv(path):
    start_time = time()

    # meta = pd.read_csv(path, header=0, sep=";", nrows=2, index_col=[0])
    # meta = reduce(meta)
    # dtypes = dict(meta.dtypes)

    data = pd.read_csv(path, header=0, sep=";", index_col=[0])

    print(f"Elapsed time: {time() - start_time} seconds")
    print("\n")
    print(data.info(verbose=False, memory_usage="deep"))
    print("\n")

    X = data.loc[:, data.columns != 'atd']
    y = data['atd'] - data['etd']

    return data, X, y


# Import data
raw, X_raw, y_raw = load_csv(RAW_PATH)
crafted, X_crafted, y_crafted = load_csv(CF_PATH)


def best_iteration(evals_result):
    best = evals_result[0]
    for i in evals_result:
        if best > i:
            best = i
    return best


def ensemble_train(X_train, X_test, y_train, y_test, params, path_metric=None, path_importance=None):
    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_test, y_test)

    evals_result = {}

    pp.pprint(params)

    bst = lgb.train(
        params,
        train_set=train_set,
        valid_sets=[valid_set],
        valid_names=["Validation error", "Train error"],
        verbose_eval=1,
        evals_result=evals_result
    )
    if path_metric is None:
        pass
    else:
        ax_metric = lgb.plot_metric(evals_result, metric="l2", ylabel="MSE", grid=False)
        plt.savefig(path_metric)
    if path_importance is None:
        pass
    else:
        ax_importance = lgb.plot_importance(bst, importance_type="split", max_num_features=15)
        plt.savefig(path_importance)

    return bst, evals_result


def lr_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    mse = mean_squared_error(y_test, lr.predict(X_test))
    print(f"Validation error: {mse}")


gbdt_optim_raw = {
    "boosting_type": "gbdt",
    "metric": "l2",
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 53,
    "learning_rate": 0.031,
    "feature_fraction": 0.65,
    "feature_fraction_bynode": 0.94,
    "num_leaves": 160,
    "min_child_samples": 205,
    "subsample_freq": 6,
    "subsample": 0.95,
    "max_bin": 1000,
    "num_threads": 6,
    "random_state": 42,
    "force_row_wise": True,
    "early_stopping": 20,
}

gbdt_optim_crafted = {
    "boosting_type": "gbdt",
    "metric": "l2",
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 52,
    "learning_rate": 0.03,
    "feature_fraction": 0.65,
    "feature_fraction_bynode": 0.94,
    "num_leaves": 160,
    "min_child_samples": 205,
    "subsample_freq": 5,
    "subsample": 0.94,
    "max_bin": 1000,
    "num_threads": 6,
    "random_state": 42,
    "force_row_wise": True,
    "early_stopping": 20,

}

rf_optim_crafted = {
    "boosting_type": "rf",
    "metric": "l2",
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 53,
    "feature_fraction": 0.82,
    "feature_fraction_bynode": 0.82,
    "subsample": 0.93,
    "subsample_freq": 6,
    "num_leaves": 1050,
    "min_child_samples": 1,
    "max_bin": 1000,
    "min_data_in_bin": 1,
    "num_threads": 6,
    "random_state": 42,
    "force_row_wise": True,
    "early_stopping": 20,
}

rf_optim_raw = {
    "boosting_type": "rf",
    "metric": "l2",
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 52,
    "feature_fraction": 0.89,
    "feature_fraction_bynode": 0.69,
    "subsample": 0.99,
    "subsample_freq": 5,
    "num_leaves": 1050,
    "min_child_samples": 1,
    "max_bin": 1000,
    "min_data_in_bin": 1,
    "num_threads": 6,
    "random_state": 42,
    "force_row_wise": True,
    "early_stopping": 20,
}


def add_noise(X, features, mu, sigma):
    np.random.seed(42)
    new_X = pd.DataFrame()

    for f in X.columns:
        if f in features:
            new_X[f] = X[f] + np.random.normal(mu, sigma)
        else:
            new_X[f] = X[f]
    return new_X


def noise_test(X, y, mu_sigma, callback, params=None, sample_size=200000):
    results = []
    selected = ["etd", "order_time", "restaurant_queue"]

    standardized_X = (X - X.mean(axis=0)) / X.std(axis=0)
    if standardized_X.isna().values.any():
        standardized_X = standardized_X.fillna(0)

    for n in mu_sigma:
        print(f"mu: {n[0]}")
        print(f"sigma: {n[1]}")

        X_train, X_test, y_train, y_test = train_test_split(standardized_X, y, train_size=0.8, random_state=42)
        X_noisy = add_noise(X_train, selected, n[0], n[1])

        if params is None:
            model = callback(X_train[:sample_size], y_train[:sample_size])
            mse = mean_squared_error(y_test, model.predict(X_test))
        else:
            model, evals_result = callback(
                X_noisy, X_test,
                y_train, y_test,
                params
            )
            mse = best_iteration(evals_result["Validation error"]["l2"])
            results.append(mse)
    plt.plot([n[1] for n in mu_sigma], results)
    plt.xlabel("Sigma")
    plt.ylabel("MSE")
    plt.show()


mu_sigma = [
    (0, 0.1),
    (0, 0.2),
    (0, 0.3),
    (0, 0.4),
    (0, 0.5),
    (0, 0.6),
    (0, 0.7),
    (0, 0.8),
    (0, 0.9),
    (0, 1.0),
]

noise_test(X_crafted, y_crafted, mu_sigma, ensemble_train, gbdt_optim_raw)
