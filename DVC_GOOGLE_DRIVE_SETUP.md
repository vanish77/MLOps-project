# Настройка Google Drive для DVC Remote

## Вариант 1: Google Drive Desktop (Рекомендуется)

### Шаг 1: Установка Google Drive Desktop

1. Скачайте и установите Google Drive Desktop:
   - https://www.google.com/drive/download/

2. Войдите в свой Google аккаунт

3. Выберите папку для синхронизации (например, `~/Google Drive`)

### Шаг 2: Настройка DVC Remote

```bash
# Создайте папку для DVC в Google Drive
mkdir -p ~/Google\ Drive/MLOps-DVC-Storage

# Настройте DVC remote на эту папку
dvc remote add gdrive ~/Google\ Drive/MLOps-DVC-Storage

# Или используйте абсолютный путь
dvc remote add gdrive "/Users/ivanevgenyevich/Google Drive/MLOps-DVC-Storage"

# Сделайте gdrive default remote
dvc remote default gdrive

# Проверьте конфигурацию
dvc remote list
```

### Шаг 3: Копирование данных

```bash
# Скопируйте существующие данные в Google Drive
cp -r dvc_storage/* ~/Google\ Drive/MLOps-DVC-Storage/

# Или переместите (если хотите использовать только Google Drive)
# mv dvc_storage/* ~/Google\ Drive/MLOps-DVC-Storage/
```

### Шаг 4: Push в Google Drive

```bash
# Отправьте все данные в Google Drive
dvc push

# Проверьте, что данные синхронизированы
# (Google Drive Desktop автоматически загрузит файлы в облако)
```

---

## Вариант 2: Rclone (Продвинутый)

### Шаг 1: Установка rclone

```bash
# macOS
brew install rclone

# Или через pip
pip install rclone
```

### Шаг 2: Настройка rclone для Google Drive

```bash
# Запустите конфигурацию
rclone config

# Выберите:
# n) New remote
# name: gdrive
# Storage: Google Drive (drive)
# Следуйте инструкциям для авторизации
```

### Шаг 3: Настройка DVC Remote

```bash
# Создайте папку в Google Drive через rclone
rclone mkdir gdrive:MLOps-DVC-Storage

# Настройте DVC remote
dvc remote add gdrive gdrive:MLOps-DVC-Storage

# Сделайте default
dvc remote default gdrive
```

### Шаг 4: Push данных

```bash
# Копируйте данные через rclone
rclone copy dvc_storage/ gdrive:MLOps-DVC-Storage/ -P

# Или используйте DVC push
dvc push
```

---

## Проверка работы

```bash
# Проверьте remote
dvc remote list

# Проверьте статус
dvc status

# Попробуйте pull (должен работать из Google Drive)
dvc pull
```

## Преимущества Google Drive

- ? Автоматическая синхронизация (Desktop)
- ? Бесплатное хранилище (15 ГБ)
- ? Доступ из любого места
- ? Простота настройки
- ? Совместный доступ (можно дать доступ коллегам)

## Ограничения

- ?? Google Drive Desktop может быть медленнее, чем S3
- ?? Ограничение на размер файла (до 5 ТБ, но рекомендуется до 100 ГБ)
- ?? Для больших проектов лучше использовать S3/GCS
