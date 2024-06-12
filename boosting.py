from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        """
        Обучает новую базовую модель и добавляет ее в ансамбль.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.
        y : array-like, форма (n_samples,)
            Массив целевых значений.
        predictions : array-like, форма (n_samples,)
            Предсказания текущего ансамбля.

        Примечания
        ----------
        Эта функция добавляет новую модель и обновляет ансамбль.
        """
        # Генерация бутстрап-выборки
        n_samples = x.shape[0]
        sample_size = int(n_samples * self.subsample)
        x_sample, y_sample, predictions_sample = resample(x, y, predictions, n_samples=sample_size)

        # Остатки
        residuals = y_sample - self.sigmoid(predictions_sample)

        # Обучение базовой модели на остатках
        model = self.base_model_class(**self.base_model_params)
        model.fit(x_sample, residuals)

        # Предсказания новой модели
        new_predictions = model.predict(x)

        # Оптимизация гаммы
        gamma = self.find_optimal_gamma(y, predictions, new_predictions)

        self.models.append(model)
        self.gammas.append(gamma)

        return gamma

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Обучает модель на тренировочном наборе данных и выполняет валидацию на валидационном наборе.

        Параметры
        ----------
        x_train : array-like, форма (n_samples, n_features)
            Массив признаков для тренировочного набора.
        y_train : array-like, форма (n_samples,)
            Массив целевых значений для тренировочного набора.
        x_valid : array-like, форма (n_samples, n_features)
            Массив признаков для валидационного набора.
        y_valid : array-like, форма (n_samples,)
            Массив целевых значений для валидационного набора.
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for i in range(self.n_estimators):
            gamma = self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.learning_rate * gamma * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * gamma * self.models[-1].predict(x_valid)

            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if self.early_stopping_rounds is not None:
                self.validation_loss[i % self.early_stopping_rounds] = valid_loss
                if (i >= self.early_stopping_rounds and
                        valid_loss > self.validation_loss.min()):
                    print(f"Early stopping after {i + 1} rounds.")
                    break

        if self.plot:
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['valid_loss'], label='Validation Loss')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Score')
            plt.legend()
            plt.show()

    def predict_proba(self, x):
        """
        Вычисляет вероятности принадлежности классу для каждого образца.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.

        Возвращает
        ----------
        probabilities : array-like, форма (n_samples, n_classes)
            Вероятности для каждого класса.
        """
        predictions = np.zeros(x.shape[0])

        for gamma, model in zip(self.gammas, self.models):
            predictions += self.learning_rate * gamma * model.predict(x)

        prob_1 = self.sigmoid(predictions)
        prob_0 = 1 - prob_1
        return np.vstack([prob_0, prob_1]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        """
        Находит оптимальное значение гаммы для минимизации функции потерь.

        Параметры
        ----------
        y : array-like, форма (n_samples,)
            Целевые значения.
        old_predictions : array-like, форма (n_samples,)
            Предыдущие предсказания ансамбля.
        new_predictions : array-like, форма (n_samples,)
            Новые предсказания базовой модели.

        Возвращает
        ----------
        gamma : float
            Оптимальное значение гаммы.

        Примечания
        ----------
        Значение гаммы определяется путем минимизации функции потерь.
        """
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        """
        Возвращает важность признаков в обученной модели.

        Возвращает
        ----------
        importances : array-like, форма (n_features,)
            Важность каждого признака.

        Примечания
        ----------
        Важность признаков определяется по вкладу каждого признака в финальную модель.
        """
        total_importances = np.zeros(self.models[0].n_features_)

        for model in self.models:
            total_importances += model.feature_importances_

        return total_importances / len(self.models)
