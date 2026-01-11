# Решение проблемы с файлами в Google Drive

## Проблема
Файлы появляются в разброс в Google Drive, а не в папке `MLOps-DVC-Storage`.

## Решение

### Вариант 1: Проверка через веб-интерфейс Google Drive

1. Откройте https://drive.google.com
2. Найдите папку `MLOps-DVC-Storage` в списке файлов
3. Если папка есть, но файлы разбросаны:
   - Выберите все файлы, которые должны быть в папке
   - Переместите их в папку `MLOps-DVC-Storage` (перетащите или используйте "Переместить")

### Вариант 2: Создание папки через веб-интерфейс

1. Откройте https://drive.google.com
2. Нажмите "Создать" ? "Папка"
3. Назовите папку: `MLOps-DVC-Storage`
4. Дождитесь синхронизации (папка появится в `~/GoogleDrive/MLOps-DVC-Storage`)
5. Запустите скрипт настройки заново:
   ```bash
   bash setup_gdrive_remote.sh
   ```

### Вариант 3: Проверка синхронизации Google Drive Desktop

Google Drive Desktop может синхронизировать файлы постепенно. Проверьте:

1. Откройте Google Drive Desktop (иконка в меню)
2. Проверьте статус синхронизации
3. Дождитесь завершения синхронизации (может занять несколько минут для 894 МБ)

### Вариант 4: Ручное перемещение файлов

Если файлы уже в Google Drive, но не в нужной папке:

1. В веб-интерфейсе Google Drive найдите все файлы из `dvc_storage`
2. Создайте папку `MLOps-DVC-Storage` (если еще нет)
3. Переместите все файлы в эту папку
4. Обновите DVC remote:
   ```bash
   dvc remote remove gdrive
   dvc remote add -d gdrive ~/GoogleDrive/MLOps-DVC-Storage
   ```

## Проверка правильности настройки

```bash
# Проверьте DVC remote
dvc remote list

# Должно быть:
# gdrive  /Users/ivanevgenyevich/GoogleDrive/MLOps-DVC-Storage  (default)

# Проверьте структуру папки
ls -la ~/GoogleDrive/MLOps-DVC-Storage/

# Должна быть папка files/ с подпапками md5/
```

## Если проблема сохраняется

1. Убедитесь, что Google Drive Desktop запущен и синхронизирован
2. Проверьте, что папка `MLOps-DVC-Storage` существует в веб-интерфейсе
3. Попробуйте перезапустить Google Drive Desktop
4. Используйте локальное хранилище как backup:
   ```bash
   dvc remote default storage
   ```
