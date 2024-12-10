import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Загружаем данные из текстового файла
data = pd.read_csv('Ирисы.txt', delimiter='\t', encoding='windows-1251')

# Заменяем запятые на точки в числовых данных
data = data.replace(',', '.', regex=True)

# Разделение данных на признаки (X) и метки (y)
X = data.iloc[:, :-1]  # Все столбцы, кроме последнего — это признаки
y = data.iloc[:, -1]   # Последний столбец — метки классов

# Преобразуем все числовые данные к float
X = X.astype(float)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Функция для оценки модели kNN
def evaluate_knn():
    # Определяем параметры для перебора по сетке
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],  # Количество соседей
        'weights': ['uniform', 'distance'],  # Способы взвешивания
        'metric': ['euclidean', 'manhattan', 'minkowski']  # Метрики расстояния
    }

    # Создаем модель kNN
    knn_classifier = KNeighborsClassifier()

    # Настраиваем перебор по сетке
    grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid,
                               scoring='accuracy', cv=5, n_jobs=-1)

    # Обучаем модель
    grid_search.fit(X_train, y_train)

    # Получаем наилучшие параметры и модель
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Предсказания для обучающей выборки
    y_train_pred = best_model.predict(X_train)
    # Метрики для обучающей выборки
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)

    # Предсказания для тестовой выборки
    y_test_pred = best_model.predict(X_test)
    # Метрики для тестовой выборки
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    # Выводим результаты
    print("Наилучшие параметры:", best_params)
    print("\nМетрики на обучающей выборке:")
    print(f"Общая точность: {train_accuracy:.2f}")
    print(f"Средняя точность (Precision): {train_precision:.2f}")
    print(f"Средняя полнота (Recall): {train_recall:.2f}")
    print(f"Средняя F1-метрика: {train_f1:.2f}")

    print("\nМетрики на тестовой выборке:")
    print(f"Общая точность: {test_accuracy:.2f}")
    print(f"Средняя точность (Precision): {test_precision:.2f}")
    print(f"Средняя полнота (Recall): {test_recall:.2f}")
    print(f"Средняя F1-метрика: {test_f1:.2f}")

    print("\nОтчет по классификации для тестовой выборки:\n", classification_report(y_test, y_test_pred))
    print("=" * 50)

# Запускаем оценку модели kNN
evaluate_knn()
