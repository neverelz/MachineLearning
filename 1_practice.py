import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Загрузка данных из Excel файла
data = pd.read_excel('practice_1.xlsx')

# Изменение к данных для студенческого 22Б1240.
data['x'] = data['x'] + 4
data['y'] = data['y'] + 1
data['z'] = data['z'] + (4 + 1) / 2

# Создание фигуры для нескольких диаграмм
fig, schema = plt.subplots(2, 2, figsize=(14, 12))

# Диаграмма 1: линейные графики для x, y и z
schema[0, 0].plot(data['Периоды времени'], data['x'], marker='.', color='r', label='x')
schema[0, 0].plot(data['Периоды времени'], data['y'], marker='.', color='g', label='y')
schema[0, 0].plot(data['Периоды времени'], data['z'], marker='.', color='b', label='z')
# Настройка графика
schema[0, 0].set_title('Графический анализ исходных данных')
schema[0, 0].set_xlabel('Год')
schema[0, 0].set_ylabel('тыс. шт')
schema[0, 0].set_xticks(ticks=data['Периоды времени'])
schema[0, 0].tick_params(axis='x', rotation=45)
schema[0, 0].legend()
schema[0, 0].grid(True)

# Вычисление матрицы корреляции
correlation_matrix = data[['x', 'y', 'z']].corr()
# Диаграмма 2: тепловая карта корреляции
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=schema[0, 1])
schema[0, 1].set_title('Матрица корреляции')

# Подготовка данных для линейной регрессии
years = data[['Периоды времени']]
X = data['x']
Y = data['y']
Z = data['z']

# Линейная регрессия для x, y и z
prediction_x = LinearRegression().fit(years, X).predict(years)
prediction_y = LinearRegression().fit(years, Y).predict(years)
prediction_z = LinearRegression().fit(years, Z).predict(years)

# Квадратичная модель тренда
polynomial_features = PolynomialFeatures(degree=2)
poly_x = polynomial_features.fit_transform(years)
quadratic_model_x = LinearRegression().fit(poly_x, X).predict(poly_x)
poly_y = polynomial_features.fit_transform(years)
quadratic_model_y = LinearRegression().fit(poly_y, Y).predict(poly_y)
poly_z = polynomial_features.fit_transform(years)
quadratic_model_z = LinearRegression().fit(poly_z, Z).predict(poly_z)

# Линейно предсказанные данные для x, y и z
schema[1, 0].plot(data['Периоды времени'], prediction_x, marker='.', color='r', linewidth=2, label='Прогноз для x')
schema[1, 0].plot(data['Периоды времени'], prediction_y, marker='.', color='g', linewidth=2, label='Прогноз для y')
schema[1, 0].plot(data['Периоды времени'], prediction_z, marker='.', color='b', linewidth=2, label='Прогноз для z')

schema[1, 1].plot(data['Периоды времени'], quadratic_model_x, marker='.', color='r', label='Квадратичный тренд для x')
schema[1, 1].plot(data['Периоды времени'], quadratic_model_y, marker='.', color='g', label='Квадратичный тренд для Y')
schema[1, 1].plot(data['Периоды времени'], quadratic_model_z, marker='.', color='b', label='Квадратичный тренд для Z')

# Настройка графика
schema[1, 0].set_title('Модель линейного тренда')
schema[1, 0].set_xlabel('Периоды времени')
schema[1, 0].set_ylabel('Значение')
schema[1, 0].set_xticks(ticks=data['Периоды времени'])
schema[1, 0].tick_params(axis='x', rotation=45)
schema[1, 0].legend()
schema[1, 0].grid(True)

# Настройка графика
schema[1, 1].set_title('Модель квадратичного тренда')
schema[1, 1].set_xlabel('Периоды времени')
schema[1, 1].set_ylabel('Значение')
schema[1, 1].set_xticks(ticks=data['Периоды времени'])
schema[1, 1].tick_params(axis='x', rotation=45)
schema[1, 1].legend()
schema[1, 1].grid(True)

# Отображение графиков
plt.tight_layout()
plt.show()


# Функция для вычисления ошибок MSE и MAE
def calculate_errors(real_values, predicted_values, name):
    mse = mean_squared_error(real_values, predicted_values)
    mae = mean_absolute_error(real_values, predicted_values)
    print(f"{name}: MSE = {mse}, MAE = {mae}")


# Расчет ошибок для линейных моделей
calculate_errors(X, prediction_x, 'Линейная модель x')
calculate_errors(Y, prediction_y, 'Линейная модель y')
calculate_errors(Z, prediction_z, 'Линейная модель z')

# Расчет ошибок для квадратичных моделей
calculate_errors(X, quadratic_model_x, 'Квадратичная модель x')
calculate_errors(Y, quadratic_model_y, 'Квадратичная модель y')
calculate_errors(Z, quadratic_model_z, 'Квадратичная модель z')