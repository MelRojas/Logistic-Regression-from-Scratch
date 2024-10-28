import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from customLogisticRegression import CustomLogisticRegression
from sklearn.linear_model import LogisticRegression


def z_standarization(df):
    scaled_data = df.copy()
    for col in scaled_data.columns:
        scaled_data[col] = (scaled_data[col] - scaled_data[col].mean()) / scaled_data[col].std()

    return scaled_data


def calc_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


n_epoch = 1000  # although this parameter can be specified only for custom models
fit_intercept = True
l_rate = 0.01
columns = ['worst concave points', 'worst perimeter', 'worst radius']  # same as in the previous stage

cut_off = 0.5


X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X = z_standarization(X[columns])
y = y.to_frame()


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

lr_mse = CustomLogisticRegression(fit_intercept, l_rate, n_epoch)
lr_mse.fit_mse(X_train.values, y_train.values.flatten())
y_mse_pred = lr_mse.predict(X_test.values, cut_off)
mse_accuracy = accuracy_score(y_test.values.flatten(), y_mse_pred)

lr_log_loss = CustomLogisticRegression(fit_intercept, l_rate, n_epoch)
lr_log_loss.fit_log_loss(X_train.values, y_train.values.flatten())
y_pred_log_loss = lr_log_loss.predict(X_test.values, cut_off)
logloss_accuracy = accuracy_score(y_test.values.flatten(), y_pred_log_loss)

lr = LogisticRegression()
lr.fit(X_train.values, y_train.values.flatten())
y_pred = lr.predict(X_test.values)
sklearn_accuracy = accuracy_score(y_test.values.flatten(), y_pred)

# output = {"coef_": lr.coef_.tolist(), "accuracy": accuracy}
# print(output)
output = {'mse_accuracy': mse_accuracy, 'logloss_accuracy': logloss_accuracy, 'sklearn_accuracy': sklearn_accuracy, 'mse_error_first': lr_mse.errors[0].flatten().tolist(), 'mse_error_last': lr_mse.errors[-1].flatten().tolist(), 'logloss_error_first': lr_log_loss.errors[0].flatten().tolist(), 'logloss_error_last': lr_log_loss.errors[-1].flatten().tolist()}
print(output)

def calc_range(before, after):
    range_before = before.max() - before.min()
    range_after = after.max() - after.min()
    return "narrowed" if range_after < range_before else "expanded"

print(f'''Answers to the questions:
1) {lr_mse.errors[0].min():.5f}
2) {lr_mse.errors[-1].min():.5f}
3) {lr_log_loss.errors[0].max():.5f}
4) {lr_log_loss.errors[-1].max():.5f}
5) {calc_range(lr_mse.errors[0], lr_mse.errors[-1])}
6) {calc_range(lr_log_loss.errors[0], lr_log_loss.errors[-1])}''')



