import os
import sys
import logging
import shutil
from datetime import datetime, timedelta
from django.conf import settings
from django.utils import timezone
import django
from celery import Celery

# Настройка пути к вашему Django-проекту
PROJECT_PATH = '/mnt/ks/Works/railcars/railcars_new/train_project'
sys.path.append(PROJECT_PATH)

# Установка переменной окружения для настроек Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'train_project.settings')

# Инициализация Django
django.setup()

# Импорт моделей
from train_app.models import Train, TrainDetail

# Настройка логирования
logging.basicConfig(
    filename='download.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

# Celery app configuration
app = Celery('railcars_automatization')
app.config_from_object('railcars_automatization.celeryconfig')

def create_traindetail(train, serial_number, image_link):
    """Создание нового TrainDetail без проверки уникальности serial_number."""
    try:
        train_detail = TrainDetail.objects.create(
            train=train,
            serial_number=serial_number,
            image_link=image_link,
        )
        logging.info(f'Создан TrainDetail для поезда {train.train_id}: {serial_number}')
    except Exception as e:
        logging.error(f'Ошибка при создании TrainDetail: {e}')

def generate_unique_train_id():
    """Генерация уникального train_id на основе максимального train_id в базе данных."""
    try:
        max_train = Train.objects.all().order_by('-train_id').first()
        return max_train.train_id + 1 if max_train else 1  # Начать с 1, если это первый поезд
    except Exception as e:
        logging.error(f'Ошибка при генерации уникального train_id: {e}')
        return None

def save_image_to_new_folder(image_path):
    """Перемещаем изображение в папку media и возвращаем относительный путь."""
    new_image_folder = '/mnt/ks/Works/railcars/railcars_new/train_project/media'
    
    if not os.path.exists(new_image_folder):
        try:
            os.makedirs(new_image_folder)
            logging.info(f"Создана новая папка для сохранения изображений: {new_image_folder}")
        except Exception as e:
            logging.error(f"Ошибка при создании папки {new_image_folder}: {e}")
            return None

    new_image_path = os.path.join(new_image_folder, os.path.basename(image_path))
    
    try:
        shutil.move(image_path, new_image_path)
        logging.info(f"Изображение перемещено в: {new_image_path}")
    except Exception as e:
        logging.error(f"Ошибка при перемещении изображения {image_path}: {e}")
        return None

    # Возвращаем относительный путь
    return f"/media/{os.path.basename(new_image_path)}"

@app.task
def import_data_to_db():
    logging.info("Старт обработки OCR результатов.")
    
    # Получаем предыдущую дату
    previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    image_folder = f'/mnt/ks/Works/railcars/railcars_new/images/{previous_day}'
    text_folder = f'/mnt/ks/Works/railcars/railcars_new/results/{previous_day}'

    if not os.path.exists(text_folder):
        logging.error(f"Text folder {text_folder} не найден.")
        return

    try:
        files = os.listdir(text_folder)
        logging.info(f"Найдено файлов: {len(files)}. Список файлов: {files}")
    except Exception as e:
        logging.error(f"Ошибка при чтении папки {text_folder}: {e}")
        return

    if len(files) == 0:
        logging.warning(f"В папке {text_folder} нет файлов для обработки.")
        return

    logging.info(f"Начало обработки OCR файлов в папке {text_folder}")

    # Генерируем уникальный train_id
    train_id = generate_unique_train_id()
    if train_id is None:
        logging.error("Не удалось сгенерировать уникальный train_id. Завершение процесса.")
        return

    # Создаем новый объект Train
    try:
        train = Train.objects.create(
            train_id=train_id,
            location='Unknown',
            direction='Unknown',
            date=(timezone.now() - timedelta(days=1)).date(),
            time=timezone.now().strftime('%H:%M:%S'),
        )
        logging.info(f"Создан поезд с train_id: {train.train_id}, ID: {train.id}")
    except Exception as e:
        logging.error(f'Ошибка при создании поезда: {e}')
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    for filename in files:
        if filename.lower().endswith('.txt'):
            logging.info(f"Обработка файла: {filename}")
            text_path = os.path.join(text_folder, filename)

            if os.path.getsize(text_path) == 0:
                logging.warning(f"Файл {text_path} пуст, пропускаем.")
                continue

            try:
                with open(text_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
            except Exception as e:
                logging.error(f"Ошибка при чтении файла {text_path}: {e}")
                continue

            # Удаляем пустые строки и получаем распознанный номер
            stripped_lines = [line.strip() for line in lines if line.strip()]
            if not stripped_lines:
                logging.warning(f"Файл {text_path} не содержит непустых строк, пропускаем.")
                continue

            # Получаем распознанный номер (допустим, это самая длинная строка)
            recognized_number = max(stripped_lines, key=len)
            logging.info(f"Распознанный номер: {recognized_number}")

            base_filename = os.path.splitext(filename)[0]
            image_found = False
            image_path = None

            # Ищем соответствующее изображение
            for ext in image_extensions:
                potential_image = os.path.join(image_folder, base_filename + ext)
                if os.path.exists(potential_image):
                    image_path = potential_image
                    image_found = True
                    break

            if not image_found:
                logging.warning(f"Изображение для {filename} не найдено, пропускаем.")
                continue

            logging.info(f"Изображение найдено: {image_path}")

            # Загружаем в базу данных только те изображения, для которых распознан номер
            if recognized_number.lower() != "не распознано":
                new_image_path = save_image_to_new_folder(image_path)
                if new_image_path:
                    create_traindetail(
                        train=train,
                        serial_number=recognized_number,
                        image_link=new_image_path,
                    )
                else:
                    logging.error(f"Не удалось переместить изображение {image_path}.")

    logging.info(f"Обработка OCR файлов завершена. Все данные сохранены для поезда с train_id {train.train_id}.")

# Запуск задачи в Celery
if __name__ == "__main__":
    import_data_to_db.apply_async()
