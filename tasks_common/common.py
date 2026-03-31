import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def compute_metrics(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, np.sqrt(mse), r2


def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(np.array(X_train, dtype=float), np.array(y_train, dtype=float))
    return reg.predict(np.array(X_test, dtype=float))


def kf_predict(embs, labels, n_splits=10):
    kf = KFold(n_splits=n_splits)
    y_preds = []
    y_truths = []

    for train_index, test_index in kf.split(embs):
        X_train, X_test = embs[train_index], embs[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_pred = regression(X_train, y_train, X_test, alpha=1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def predict_regression(embs, labels, display=False):
    y_pred, y_true = kf_predict(embs, labels)
    mae, rmse, r2 = compute_metrics(y_pred, y_true)
    if display:
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2:   {r2:.3f}")
    return mae, rmse, r2

