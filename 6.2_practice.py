import pandas as pd

# Загрузка данных
data = pd.read_csv("Supermarket.txt", sep="\t", encoding="windows-1251", names=["Чек", "Товар"])

# Находим чеки, содержащие мед и сыры
honey_receipts = set(data[data["Товар"] == "МЕД"]["Чек"])
cheese_receipts = set(data[data["Товар"] == "СЫРЫ"]["Чек"])
common_receipts = honey_receipts & cheese_receipts  # Чеки, где есть и мед, и сыры

# Собираем товары из этих чеков, исключая мед и сыры
filtered_data = data[data["Чек"].isin(common_receipts) & ~data["Товар"].isin(["МЕД", "СЫРЫ"])]

# Подсчитываем частоту товаров
item_counts = filtered_data["Товар"].value_counts()

# Выводим товары, которые реже всего встречаются
rare_items = item_counts[item_counts == item_counts.min()]
print("Товары, которые реже всего встречаются с медом и сырами:")
print(rare_items)
