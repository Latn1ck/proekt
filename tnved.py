import pandas as pd


def join_categories(group):
    strings = group.tolist()
    first = strings[0]
    # Извлекаем категорию из первой строки
    category = first.split(':')[0] + ': '
    
    result = [first]
    for s in strings[1:]:
        if s.startswith(category):
            result.append(s[len(category):])
        else:
            result.append(s)
    
    return ' '.join(result)

df=pd.read_excel('TWS_TNVED_2026-05-18.xlsx',sheet_name='ТНВЭД')
df['Наименование'] = df['Наименование'].str.replace(' 🠺', '', regex=False)
df['Код6']=df['Код'].astype(str).str[:6]
del df['Подробности']
del df['Тариф']
result = df.groupby('Код6')['Наименование'].agg(join_categories).reset_index()
dict={k:v for k,v in zip(result['Код6'],result['Наименование'])}