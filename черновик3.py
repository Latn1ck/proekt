from transformers import BertTokenizer, BertModel, BertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
import chernovik as ch
import pandas as pd
import chernovik2 as ch2
import torch
from sklearn.model_selection import train_test_split

def chapterPreprocess(x):
    if x[:6]=="РАЗДЕЛ":
        x=x[10:]
    x=x.replace('\n',' ')
    x=x.replace('\t','')
    return x.lower()
def get_class_embeddings(class_dict):
    embeddings = {}
    with torch.no_grad():
        for class_id, desc in class_dict.items():
            inputs = tokenizer(
                desc, 
                return_tensors='pt',
                max_length=256,
                padding='max_length',
                truncation=True
            )
            outputs = bert_model(**inputs)
            embeddings[class_id] = outputs.last_hidden_state.mean(dim=1)
    return embeddings
def accuracy(pred,ground_truth):
    return (np.sum(pred==ground_truth))/(len(pred))
#подгрузка датасета
X_train=ch2.X_train
X_test=ch2.X_test
y_train=ch2.y_train
y_test=ch2.y_test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
dictOKRBexpl=ch.dictOKRBexpl
dictOKRBklass=ch.dictOKRBklass
shortDictOKRB={}
for i in dictOKRBexpl.keys():
    if not dictOKRBexpl[i]=='':
        shortDictOKRB[i]=chapterPreprocess(dictOKRBexpl[i])
    else:
        shortDictOKRB[i]=chapterPreprocess(dictOKRBklass[i])
shortDictOKRB={k:v for k,v in zip(shortDictOKRB.keys(),shortDictOKRB.values()) if len(k)==2}
shortDictOKRB['00']=''
shortDictOKRB=dict(sorted(shortDictOKRB.items()))
print(shortDictOKRB)
#инициализация модели
tokenizer = BertTokenizer.from_pretrained('sberbank-ai/ruBert-base')
bert_model = BertModel.from_pretrained('sberbank-ai/ruBert-base')
model = BertForSequenceClassification.from_pretrained(
    "sberbank-ai/ruBert-base",
    num_labels=len(shortDictOKRB.keys()),
    problem_type="single_label_classification"
)
#эмбеддинги описаний классов
OKRBEmbeddings = {}
with torch.no_grad():
    for class_id, desc in shortDictOKRB.items():
        inputs = tokenizer(desc, return_tensors='pt',max_length=256,padding='max_length',truncation=True)
        outputs = bert_model(**inputs)
        OKRBEmbeddings[class_id] = outputs.last_hidden_state.mean(dim=1)
#fine tuning
class EnhancedBertClassifier(nn.Module):
    def __init__(self, bert_model, class_embeddings):
        super().__init__()
        self.bert = bert_model
        self.class_embeddings = nn.ParameterDict({
            str(k): nn.Parameter(v.clone().detach()) for k, v in class_embeddings.items()
        })
        self.classifier = nn.Linear(768 * 2, len(class_embeddings))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        # Получаем эмбеддинг товара
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        product_embedding = outputs.last_hidden_state[:, 0, :]  # Используем [CLS] токен
        
        # Сравниваем с каждым классом
        logits = []
        for class_id in self.class_embeddings.keys():
            class_embed = self.class_embeddings[class_id]
            # Расширяем эмбеддинг класса для батча
            expanded_class_embed = class_embed.expand(product_embedding.size(0), -1)
            combined = torch.cat([product_embedding, expanded_class_embed], dim=1)
            combined = self.dropout(combined)
            logits.append(self.classifier(combined))
        
        return torch.stack(logits, dim=1)  # [batch_size, num_classes, num_classes]
enhanced_model = EnhancedBertClassifier(bert_model, OKRBEmbeddings)
def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Получаем логиты для всех классов
            all_logits = model(input_ids, attention_mask)
            
            # Выбираем только соответствующие меткам логиты
            logits = all_logits[torch.arange(all_logits.size(0)), labels]
            
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Валидация
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            all_logits = model(input_ids, attention_mask)
            logits = all_logits[torch.arange(all_logits.size(0)), labels]
            
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
    
    return total_loss/len(data_loader), correct/len(data_loader.dataset)
#предикт
def predict(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            all_logits = model(input_ids, attention_mask)
            # Усредняем логиты по всем классам
            avg_logits = all_logits.mean(dim=1)
            _, preds = torch.max(avg_logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels
#результаты
train_loader = DataLoader(
    train_dataset,
    batch_size=32,      # Размер батча (16, 32, 64...)
    shuffle=True,       # Перемешивание данных перед каждой эпохой
    num_workers=2       # Количество потоков для загрузки (ускоряет процесс)
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,      # Размер батча (16, 32, 64...)
    shuffle=True,       # Перемешивание данных перед каждой эпохой
    num_workers=2       # Количество потоков для загрузки (ускоряет процесс)
)
train_model(enhanced_model, train_loader, val_loader, epochs=3)

torch.save({
    'model_state_dict': enhanced_model.state_dict(),
    'class_embeddings': OKRBEmbeddings
}, 'okrb_classifier.pth')