import numpy as np


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = None # bias = coef_[0] and weights = coef_[1:]
        self.errors = []

    def sigmoid(self, t) -> float: #[0,1]
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_): #feed_forward
        t = np.dot(row, coef_[1:]) + coef_[0] if self.fit_intercept else np.dot(row, coef_)
        # t = coef_[0] + (row @ coef_[1:])
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.coef_ = np.zeros(n_features + 1) if self.fit_intercept else np.zeros(n_features) # initialized weights

        for _ in range(self.n_epoch):
            error_list = []
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                error = y_hat - y_train[i]
                gradient = error * y_hat * (1 - y_hat)
                # update all weights
                if self.fit_intercept:
                    self.coef_[0] -= self.l_rate * gradient
                    self.coef_[1:] -= self.l_rate * gradient * row
                else:
                    self.coef_ -= (self.l_rate * gradient * row)
                error_list.append((error ** 2) * 1 / n_samples)
            self.errors.append(np.array(error_list))


    def fit_log_loss(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.coef_ = np.zeros(n_features + 1) if self.fit_intercept else np.zeros(n_features)  # initialized weights

        for _ in range(self.n_epoch):
            error_list = []
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                error = y_hat - y_train[i]
                # log_loss = -y_train[i] * np.log(y_hat) + (1 - y_train[i]) * np.log(1 - y_hat)
                log_loss = -y_train[i] * np.log(y_hat + 1e-15) - (1 - y_train[i]) * np.log(1 - y_hat + 1e-15)  # Log-loss

                # update all weights
                if self.fit_intercept:
                    self.coef_[0] -= (self.l_rate * error) / n_samples
                    self.coef_[1:] -= (self.l_rate * error * row) / n_samples
                else:
                    self.coef_ -= (self.l_rate * error * row) / n_samples
                error_list.append(log_loss * (1 / n_samples))
            self.errors.append(np.array(error_list))
    def predict(self, X_test, cut_off=0.5):
        predictions = []
        for row in X_test:
            y_hat = self.predict_proba(row, self.coef_)
            # predictions.append(y_hat)
            predictions.append(1 if y_hat >= cut_off else 0)
        return np.array(predictions)  # predictions are binary values - 0 or 1
