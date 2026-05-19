from datetime import datetime
import DatasetCreation as ch2

start=datetime.now()
print(f'Начало: {start}')
X,y=ch2.X,ch2.y
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)
#model = SentenceTransformer('Nehc/e5-large-ru')
finish=datetime.now()
def split_hierarchy(text):
    """
    Разбивает иерархический текст на уровни.
    Пример: 'Обувь: Прочая обувь: с верхом из пластмассы'
    -> ['Обувь', 'Обувь: Прочая обувь', 'Обувь: Прочая обувь: с верхом из пластмассы']
    """
    parts = text.split(':')
    hierarchies = []
    current_hierarchy = []
    for part in parts:
        if part.strip():  # Игнорируем пустые части
            current_hierarchy.append(part.strip())
            hierarchies.append(': '.join(current_hierarchy))
    return hierarchies

# Пример использования
text = "Обувь, гетры и аналогичные изделия; их детали: Прочая обувь с подошвой и с верхом из резины или пластмассы: обувь прочая: прочая: с защитным металлическим подноском"
levels = split_hierarchy(text)
print(levels)
print(f'Конец: {finish}')