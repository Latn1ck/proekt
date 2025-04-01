from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Указываем название модели (ruRoberta-large от Sberbank AI)
model_name = "ai-forever/ruRoberta-large"
# Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
text = "Я обожаю эту модель, она работает прекрасно!"
# Токенизация текста
inputs = tokenizer(
    text, 
    return_tensors="pt",  # Возвращает тензоры PyTorch
    truncation=True,      # Обрезает текст, если он длиннее максимальной длины
    padding=True,         # Добавляет padding до максимальной длины
    max_length=512        # Максимальная длина (можно уменьшить для скорости)
)
# Пример вывода inputs:
# {'input_ids': tensor([[0, 123, 456, ..., 2]]), 'attention_mask': tensor([[1, 1, 1, ..., 0]])}
with torch.no_grad():  # Отключаем вычисление градиентов (для инференса)
    outputs = model(**inputs)
# Извлекаем логиты (сырые оценки классов)
logits = outputs.logits
# Преобразуем в вероятности (softmax)
probabilities = torch.softmax(logits, dim=1).tolist()[0]
print(f"Вероятности классов: {probabilities}")
# Пример вывода: [0.02, 0.98] → 98% вероятности, что текст позитивный
