# ?? Quick Start Guide

## Установка за 3 шага

### 1. Создайте виртуальное окружение
```bash
python3 -m venv .venv
source .venv/bin/activate  # для macOS/Linux
# или
.venv\Scripts\activate     # для Windows
```

### 2. Установите зависимости
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Запустите обучение
```bash
python scripts/train.py --config configs/baseline.yaml
```

---

## Что произойдет при обучении

1. **Загрузка датасета IMDb** (~80 MB)
   - 50,000 отзывов (25k positive, 25k negative)
   - Автоматическое разбиение на train/val/test

2. **Загрузка модели DistilBERT**
   - Pretrained модель от Hugging Face
   - ~260 MB

3. **Обучение** (~15-30 минут на CPU, ~5 минут на GPU)
   - 2 эпохи (по умолчанию)
   - Batch size: 16 (train), 32 (eval)
   - Learning rate: 2e-5

4. **Результаты сохраняются в**:
   - `artefacts/distilbert-imdb/` - обученная модель
   - `artefacts/logs/train.log` - детальные логи
   - Консоль - прогресс и метрики

---

## Примеры команд

### Базовое обучение
```bash
python scripts/train.py --config configs/baseline.yaml
```

### С детальными логами
```bash
python scripts/train.py --config configs/baseline.yaml --verbose
```

### Изменение learning rate
```bash
python scripts/train.py --config configs/baseline.yaml \
  -o training.learning_rate=3e-5
```

### Изменение числа эпох и batch size
```bash
python scripts/train.py --config configs/baseline.yaml \
  -o training.num_train_epochs=3 \
  -o training.per_device_train_batch_size=8
```

### Обучение на CPU (без fp16)
```bash
python scripts/train.py --config configs/baseline.yaml \
  -o training.fp16=false
```

---

## После обучения

### Проверка метрик
```bash
cat artefacts/distilbert-imdb/metrics_test.json
```

### Валидация на примерах
```bash
python scripts/validate.py --model-path artefacts/distilbert-imdb
```

### Свои примеры
```bash
python scripts/validate.py --model-path artefacts/distilbert-imdb \
  --examples "This movie was amazing!" "Terrible film, waste of time."
```

### Использование в коде
```python
python example_inference.py
```

---

## Ожидаемые результаты

После успешного обучения вы должны увидеть примерно такие метрики:

```
Test metrics:
  accuracy: 0.9100+
  precision: 0.9000+
  recall: 0.9200+
  f1: 0.9100+
```

Целевая метрика (из README): **Accuracy ? 90%** ?

---

## Troubleshooting

### Ошибка "CUDA out of memory"
```bash
# Уменьшите batch size
python scripts/train.py --config configs/baseline.yaml \
  -o training.per_device_train_batch_size=8
```

### Медленно на CPU
```bash
# Отключите fp16 и уменьшите workers
python scripts/train.py --config configs/baseline.yaml \
  -o training.fp16=false \
  -o training.dataloader_num_workers=0
```

### Нет места на диске
```bash
# Удалите кэш датасетов (можно скачать заново)
rm -rf ~/.cache/huggingface/
```

---

## Дополнительные ресурсы

- ?? **Полная инструкция**: [SETUP.md](SETUP.md)
- ?? **Итоговая сводка**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- ?? **Описание проекта**: [README.md](README.md)

---

**Готово! Теперь можно запускать обучение** ??

