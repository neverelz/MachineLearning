import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных с указанием кодировки
data = pd.read_csv('diabetes.txt', delimiter='\t', encoding='windows-1251')

# Определение признаков и целевой переменной
X = data.drop(columns=['Диагноз'])  # Признаки
y = data['Диагноз']  # Целевая переменная

# Вычисление корреляции
correlation_matrix = X.corrwith(y).abs()  # Абсолютные значения корреляции
correlation_matrix = correlation_matrix.sort_values(ascending=False)  # Сортировка по убыванию

# Выбор наилучшего признакового пространства: исключение 2 наименее коррелирующих признаков
selected_features = correlation_matrix.index[:-2]  # Все, кроме двух последних
X_selected = X[selected_features]  # Новая матрица признаков

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)  # Нормализация признаков

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.01, random_state=42)
# Добавление столбца единиц для theta_0
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Обучающая выборка
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))  # Тестовая выборка

# Гиперпараметры
learning_rate = 0.01  # Шаг обучения
n_iterations = 1000  # Количество итераций

# Инициализация параметров модели
theta = np.zeros(X_train.shape[1])  # Вектор параметров


# Сигмоида
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Метод максимального правдоподобия (MLE) с использованием градиентного спуска
def fit_logistic_regression(X, y, theta, learning_rate, n_iterations):
    for i in range(n_iterations):
        z = np.dot(X, theta)  # Линейная комбинация признаков
        y_pred = sigmoid(z)  # Предсказанные вероятности

        # Вычисление градиентов на основе MLE
        gradient = (1 / len(y)) * np.dot(X.T, (y_pred - y))

        # Обновление параметров
        theta -= learning_rate * gradient
    return theta


# Обучение модели
theta = fit_logistic_regression(X_train, y_train, theta, learning_rate, n_iterations)

# Оценка точности
y_test_pred_prob = sigmoid(np.dot(X_test, theta))  # Предсказанные вероятности на тестовой выборке
y_test_pred = (y_test_pred_prob >= 0.5).astype(int)  # Преобразование вероятностей в классы
accuracy = np.mean(y_test_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')
