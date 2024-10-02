import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
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


# Функция для оценки различных моделей SVM
def evaluate_svm(kernel, C=1.0, degree=3, gamma='scale'):
    # Определяем параметры для перебора по сетке
    param_grid = {
        'C': [0.1, 1, 10],  # Параметр регуляризации
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Различные ядра
        'degree': [2, 3, 4],  # Степень для полиномиального ядра
        'gamma': ['scale', 'auto']  # Параметр для RBF и сигмоидного ядра
    }

    # Создаем и обучаем модель SVM с заданными параметрами
    svm_classifier = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
    # Создаем объект GridSearchCV
    grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid,
                               scoring='accuracy', cv=5, n_jobs=-1)

    # Обучаем модель с Grid Search
    grid_search.fit(X_train, y_train)

    # Получаем наилучшие параметры и результаты
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Делаем предсказания на тестовой выборке
    y_pred = best_model.predict(X_test)

    # Рассчитываем метрики для проверки качества модели с zero_division
    accuracy = accuracy_score(y_test, y_pred)  # Общая точность
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Средняя точность (Precision)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)  # Средняя полнота (Recall)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)  # Средняя F1-метрика

    # Оценка количества опорных векторов
    n_support_vectors = best_model.n_support_
    total_support_vectors = sum(n_support_vectors)

    # Выводим результаты
    print(f"Ядро: {kernel}, C: {C}, degree: {degree}, gamma: {gamma}")
    print(f"Общая точность модели: {accuracy:.2f}")
    print(f"Средняя точность (Precision): {precision:.2f}")
    print(f"Средняя полнота (Recall): {recall:.2f}")
    print(f"Средняя F1-мерa: {f1:.2f}")
    print(f"Число опорных векторов: {total_support_vectors}\n")
    print("Лучшие параметры:", best_params)
    print(f"Отчет по классификации для ядра {kernel}:\n", classification_report(y_test, y_pred))
    print("=" * 50)


# Сценарии для различных ядер и параметров
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
C_values = [0.1, 1, 10]
degrees = [2, 3, 4]  # Для полиномиального ядра
gamma_values = ['scale', 'auto']  # Для RBF и сигмоидного ядра

# Линейное ядро
for C in C_values:
    evaluate_svm(kernel='linear', C=C)

# RBF ядро
for C in C_values:
    for gamma in gamma_values:
        evaluate_svm(kernel='rbf', C=C, gamma=gamma)

# Полиномиальное ядро
for C in C_values:
    for degree in degrees:
        evaluate_svm(kernel='poly', C=C, degree=degree)

# Сигмоидное ядро
for C in C_values:
    for gamma in gamma_values:
        evaluate_svm(kernel='sigmoid', C=C, gamma=gamma)
