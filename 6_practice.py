from collections import defaultdict
from itertools import combinations
import pandas as pd


def load_data(filename):
    data = defaultdict(list)
    with open(filename, "r", encoding="cp1251") as file:
        next(file)
        for line in file:
            check, product = line.strip().split("\t")
            data[check].append(product)
    return list(data.values())


def find_frequent_itemsets(transactions, min_support):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
        for size in range(2, len(transaction) + 1):
            for combo in combinations(transaction, size):
                item_counts[frozenset(combo)] += 1
    num_transactions = len(transactions)
    return {item: count for item, count in item_counts.items() if count / num_transactions >= min_support}


def generate_rules(frequent_itemsets, min_confidence, num_transactions):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for item in itemset:
            antecedent = frozenset([item])
            consequent = itemset - antecedent
            support = frequent_itemsets[itemset] / num_transactions
            confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
            if confidence >= min_confidence:
                rules.append((antecedent, consequent, support, confidence))
    return rules


def find_best_association_with_wafer(rules):
    max_confidence = 0
    best_item = None
    for antecedent, consequent, support, confidence in rules:
        if 'ВАФЛИ' in antecedent:
            for item in consequent:
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_item = item
    return best_item, max_confidence


def find_rare_items_with_honey_and_cheese(filename):
    data = pd.read_csv(filename, sep="\t", encoding="windows-1251", names=["Чек", "Товар"])

    honey_receipts = set(data[data["Товар"] == "МЕД"]["Чек"])
    cheese_receipts = set(data[data["Товар"] == "СЫРЫ"]["Чек"])
    common_receipts = honey_receipts & cheese_receipts

    filtered_data = data[data["Чек"].isin(common_receipts) & ~data["Товар"].isin(["МЕД", "СЫРЫ"])]

    item_counts = filtered_data["Товар"].value_counts()

    rare_items = item_counts[item_counts == item_counts.min()]
    return rare_items


def find_top_itemsets(frequent_itemsets, num_top=5):
    sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: x[1], reverse=True)
    return sorted_itemsets[:num_top]


def display_results(params, frequent_itemsets, rules):
    print(
        f"| {'Параметры':<20} | {'Частых наборов':<20} | {'Правил':<10} | {'Пример правила':<30} | {'Поддержка':<10} | {'Уверенность':<10} |")
    print(f"{'-' * 100}")
    if rules:
        example_rule = rules[0]
        print(
            f"| {params:<20} | {len(frequent_itemsets):<20} | {len(rules):<10} | {set(example_rule[0])} => {set(example_rule[1]):<20} | {example_rule[2]:<10.2f} | {example_rule[3]:<10.2f} |")
    else:
        print(
            f"| {params:<20} | {len(frequent_itemsets):<20} | {len(rules):<10} | {'Нет правил':<30} | {'-':<10} | {'-':<10} |")


filename = "Supermarket.txt"

# Различные комбинации поддержки и достоверности
parameters = [
    {"min_support": 0.05, "min_confidence": 0.3},
    {"min_support": 0.1, "min_confidence": 0.5},
    {"min_support": 0.2, "min_confidence": 0.3},
    {"min_support": 0.3, "min_confidence": 0.7},
    {"min_support": 0.15, "min_confidence": 0.6},
]

# Проведение экспериментов с параметрами
transactions = load_data(filename)
num_transactions = len(transactions)

# Вывод результатов
print(f"| {'min_support':<12} | {'min_confidence':<15} | {'Правил':<8} |")
print(f"{'-' * 43}")
for params in parameters:
    frequent_itemsets = find_frequent_itemsets(transactions, params['min_support'])
    rules = generate_rules(frequent_itemsets, params['min_confidence'], num_transactions)
    print(f"| {params['min_support']:<12} | {params['min_confidence']:<15} | {len(rules):<8} |")

    # Пример 5 ассоциативных правил (если они есть)
    if rules:
        for rule in rules[:5]:
            print(f"Правило: {set(rule[0])} => {set(rule[1])}, поддержка: {rule[2]:.2f}, уверенность: {rule[3]:.2f}")
    else:
        print("Нет правил для данной комбинации параметров.")



best_item, max_confidence = find_best_association_with_wafer(rules)
if best_item:
    print(f"\nТовар с наибольшей уверенностью, связанный с вафлями: {best_item} с уверенностью {max_confidence:.2f}")
else:
    print("Нет товара с наибольшей уверенностью, связанного с вафлями.")


rare_items = find_rare_items_with_honey_and_cheese(filename)
if not rare_items.empty:
    print("\nТовары, которые реже всего встречаются с медом и сырами:")
    print(rare_items.reset_index().to_string(index=False, header=True))
else:
    print("\nНет товаров, которые реже всего встречаются с медом и сырами.")


frequent_itemsets = find_frequent_itemsets(transactions, min_support=0.1)
top_itemsets = find_top_itemsets(frequent_itemsets, num_top=5)
print("\nТоп-5 самых популярных наборов товаров:")
for itemset, count in top_itemsets:
    print(f"{set(itemset)}: встречается {count} раз")


print("\nПример ассоциативных правил:")
for i, (antecedent, consequent, support, confidence) in enumerate(rules[:5]):
    print(f"{i+1}. {set(antecedent)} => {set(consequent)} | Поддержка: {support:.2f} | Уверенность: {confidence:.2f}")
