import pandas as pd

file_path = 'Trade.txt'
df = pd.read_csv(file_path, delimiter='\t', encoding='cp1251')

def convert_date(row):
    year, month = row.split('-')
    month_number = int(month[1:])
    return f'01.{month_number:02d}.{year}'

df['Дата'] = df['Дата'].apply(convert_date)
df['Количество'] = df['Количество'].astype(int)

output_file = 'Trade.txt'
df.to_csv(output_file, sep='\t', index=False, encoding='cp1251')