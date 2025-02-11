import os
import django
import logging
from datetime import datetime
import sys
# Django setup
# Настройка пути к вашему Django-проекту
PROJECT_PATH = '/mnt/ks/Works/railcars/railcars_new/train_project'
sys.path.append(PROJECT_PATH)

# Установка переменной окружения для настроек Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'train_project.settings')

# Инициализация Django
django.setup()


from train_app.models import Train, TrainDetail

# Настройка логирования
logging.basicConfig(
    filename='update_db_changes.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_original_image_name(crop_image_name):
    """
    Преобразует имя обрезанного изображения в оригинальное.
    Например: '001_20250107164038_[M][0@0][0]_crop_0.jpg' -> '001_20250107164038_[M][0@0][0].jpg'
    """
    # Убираем '_crop_0' из имени файла
    return crop_image_name.replace('_crop_0', '')

def update_serial_numbers(ocr_results_file, train_id, date_str):
    """
    Обновляет serial_number в базе данных для конкретного поезда и даты.
    """
    updated_count = 0
    error_count = 0
    skipped_count = 0
    
    logger.info("=" * 80)
    logger.info(f"НАЧАЛО ОБНОВЛЕНИЯ")
    logger.info(f"Поезд ID: {train_id}")
    logger.info(f"Дата: {date_str}")
    logger.info(f"Файл результатов: {ocr_results_file}")
    logger.info("=" * 80)
    
    try:
        # Находим нужный поезд
        train = Train.objects.get(train_id=train_id, date=date_str)
        logger.info(f"Найден поезд: ID={train_id}, дата={date_str}, DB_id={train.id}")
    except Train.DoesNotExist:
        logger.error(f"Поезд не найден: ID={train_id}, дата={date_str}")
        return 0, 1, 0
    
    try:
        with open(ocr_results_file, 'r') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    # Разбиваем строку на компоненты
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        logger.warning(f"Строка {line_number}: Некорректный формат: {line}")
                        error_count += 1
                        continue
                        
                    image_path, serial_number, confidence = parts
                    crop_image_name = os.path.basename(image_path)
                    original_image_name = get_original_image_name(crop_image_name)
                    serial_number = serial_number.strip()
                    confidence = float(confidence)
                    
                    logger.info(f"\nОбработка строки {line_number}:")
                    logger.info(f"Обрезанное изображение: {crop_image_name}")
                    logger.info(f"Оригинальное изображение: {original_image_name}")
                    logger.info(f"Распознанный номер: {serial_number}")
                    logger.info(f"Уверенность: {confidence:.4f}")
                    
                    # Ищем запись в базе данных по оригинальному имени изображения
                    train_details = TrainDetail.objects.filter(
                        train=train,
                        image_link__contains=original_image_name
                    )
                    
                    if train_details.exists():
                        train_detail = train_details.first()
                        old_serial = train_detail.serial_number
                        
                        # Если номера разные - обновляем
                        if old_serial != serial_number:
                            train_detail.serial_number = serial_number
                            train_detail.save()
                            
                            logger.info(f"ОБНОВЛЕНО:")
                            logger.info(f"  - ID записи: {train_detail.id}")
                            logger.info(f"  - Изображение: {train_detail.image_link}")
                            logger.info(f"  - Старый номер: {old_serial}")
                            logger.info(f"  - Новый номер: {serial_number}")
                            logger.info(f"  - Уверенность OCR: {confidence:.4f}")
                            updated_count += 1
                        else:
                            logger.info(f"ПРОПУЩЕНО: номер не изменился ({old_serial})")
                            skipped_count += 1
                    else:
                        logger.warning(
                            f"НЕ НАЙДЕНО: изображение {original_image_name} "
                            f"для поезда {train_id}"
                        )
                        error_count += 1
                        
                except Exception as e:
                    logger.error(f"Ошибка в строке {line_number}: {str(e)}")
                    error_count += 1
                    continue
                    
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {ocr_results_file}: {str(e)}")
        return 0, 1, 0
        
    logger.info("\n" + "=" * 80)
    logger.info("ИТОГИ ОБНОВЛЕНИЯ:")
    logger.info(f"Всего обработано строк: {line_number}")
    logger.info(f"Обновлено записей: {updated_count}")
    logger.info(f"Пропущено (без изменений): {skipped_count}")
    logger.info(f"Ошибок: {error_count}")
    logger.info("=" * 80 + "\n")
    
    return updated_count, error_count, skipped_count


if __name__ == "__main__":
    # Параметры для обновления
    ocr_results_file = "/mnt/ks/Works/railcars/railcars_new/valid_codes/deep_text_recognition_benchmark/log_demo_result.txt"  # Путь к файлу с результатами OCR
    train_id = 95  # ID поезда для обновления
    date_str = "2025-01-29"  # Дата в формате YYYY-MM-DD
    
    updated, errors, skipped = update_serial_numbers(ocr_results_file, train_id, date_str)
    
    print(f"\nОбновление завершено:")
    print(f"- Обновлено записей: {updated}")
    print(f"- Пропущено (без изменений): {skipped}")
    print(f"- Ошибок: {errors}")
    print(f"\nПодробности смотрите в файле: update_db_changes.log")



















