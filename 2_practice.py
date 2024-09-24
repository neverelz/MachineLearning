import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка данных из файла
data = pd.read_excel('Исходные данные.xlsx')

# Разделение данных на обучающую (первые 20 строк) и тестовую выборки (последние 5 строк)
train_data = data.iloc[:20]
test_data = data.iloc[20:]

# Разделение обучающей выборки на 60% и 40%
X_train_full = train_data[['x1', 'x2', 'x3', 'x4']]
y_train_full = train_data['y']

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.4, random_state=42)

# Используем лучшие признаки
best_i, best_j = 0, 1  # Это можно улучшить
X_train_best = X_train.iloc[:, [best_i, best_j]].to_numpy()
X_val_best = X_val.iloc[:, [best_i, best_j]].to_numpy()
X_test_best = test_data[['x1', 'x2', 'x3', 'x4']].iloc[:, [best_i, best_j]].to_numpy()

# Обучение модели на выбранных признаках
model_mgua = LinearRegression()
model_mgua.fit(X_train_best, y_train)

# Предсказание на валидационной выборке
y_val_pred = model_mgua.predict(X_val_best)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f'Validation MSE: {val_mse}')

# Предсказание на тестовой выборке для модели по МГУА
y_test_pred_mgua = model_mgua.predict(X_test_best)

# Формируем таблицу для сравнения
comparison_table = pd.DataFrame({
    'Фактические значения': test_data['y'],
    'Предсказания модели МГУА': y_test_pred_mgua
})

# Вывод таблицы
print(comparison_table)
