import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
import gc
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import paramiko
import logging
from datetime import datetime, timedelta

from celery import Celery


# ===== Импортируем нужные модули из deep_text_recognition_benchmark =====
import string
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

# ВАЖНО: меняйте пути на корректные относительно вашего проекта
from railcars_automatization.deep_text_recognition_benchmark.dataset import AlignCollate
from railcars_automatization.deep_text_recognition_benchmark.dataset import RawDataset as RawDatasetBenchmark
from railcars_automatization.deep_text_recognition_benchmark.utils import CTCLabelConverter, AttnLabelConverter
from railcars_automatization.deep_text_recognition_benchmark.model import Model
# If 'modules' is a top-level package


# Celery app configuration
app = Celery('railcars_automatization')
app.config_from_object('railcars_automatization.celeryconfig')

# Logging configuration
logging.basicConfig(
    filename='automation.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

# SFTP settings
SFTP_HOST = os.getenv('SFTP_HOST', '77.95.56.87')
SFTP_PORT = int(os.getenv('SFTP_PORT', 7543))
SFTP_USERNAME = os.getenv('SFTP_USERNAME', 'rmsftp_usr')
SFTP_PASSWORD = os.getenv('SFTP_PASSWORD', '3708Ey0gXA6g')

REMOTE_BASE_FOLDER = '/rmdocfolder/image/DH-IPC-HFW3441DGP-AS-4G-NL668EA/'
LOCAL_BASE_FOLDER = '/mnt/ks/Works/railcars/railcars_new'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


# =============================================================================
# 1. Класс Config для Deep Text Recognition
# =============================================================================
class Config:
    """
    Updated Config class to match the saved model architecture
    """
    # Параметры путей
    image_folder = "/mnt/ks/Works/railcars/railcars_new/valid_codes/validation"
    saved_model = "/mnt/ks/Works/railcars/railcars_new/railcars_automatization/deep_text_recognition_benchmark/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111_1000epoch/best_accuracy.pth"

    # Параметры загрузки данных
    workers = 4
    batch_size = 1

    # Параметры, связанные с процессингом данных
    batch_max_length = 25
    imgH = 32
    imgW = 100
    rgb = False
    # Расширенный набор символов для соответствия сохраненной модели
    character = '0123456789abcdefghijklmnopqrstuvwxyz'  # 36 символов + 2 спец. символа
    sensitive = False
    PAD = False

    # Архитектура модели
    Transformation = "TPS"
    FeatureExtraction = "ResNet"
    SequenceModeling = "BiLSTM"
    Prediction = "Attn"
    num_fiducial = 20
    input_channel = 1
    output_channel = 512
    hidden_size = 256

    # Другие настройки
    num_gpu = torch.cuda.device_count()

# =============================================================================
# 2. Инициализируем модель распознавания текста (один раз)
# =============================================================================

def init_deep_text_model():
    """
    Инициализируем модель Deep Text Recognition и возвращаем:
    1) model_deep - объект PyTorch модели
    2) converter_deep - конвертер (AttnLabelConverter или CTCLabelConverter)
    3) opt_deep - конфиг, чтобы хранить нужные параметры
    """
    opt_deep = Config()

    # Если чувствительность к регистру нужна — расширяем набор символов
    if opt_deep.sensitive:
        opt_deep.character = string.printable[:-6]

    # Определяем, какой конвертер нужен (Attn или CTC)
    if 'Attn' in opt_deep.Prediction:
        converter_deep = AttnLabelConverter(opt_deep.character)

    opt_deep.num_class = len(converter_deep.character)

    # Если RGB, то меняем число каналов
    if opt_deep.rgb:
        opt_deep.input_channel = 3

    # Создаём модель
    model_deep = Model(opt_deep)
    model_deep = torch.nn.DataParallel(model_deep).to(device)

    # Загружаем веса
    logger.info(f"Loading OCR model weights from {opt_deep.saved_model}")
    model_deep.load_state_dict(torch.load(opt_deep.saved_model, map_location=device))
    model_deep.eval()

    # Настраиваем cudnn
    cudnn.benchmark = True
    cudnn.deterministic = True

    logger.info("Deep Text Recognition model initialized successfully.")
    return model_deep, converter_deep, opt_deep

# =============================================================================
# 3. Функция OCR: принимает вырезанный NumPy-фрагмент и возвращает распознанный текст
# =============================================================================
def infer_text_deep(cropped_image_np, model_deep, converter_deep, opt_deep):
    """
    Запускает инференс deep_text_recognition_benchmark на одном NumPy-изображении.
    Возвращает текст (str) и confidence (float).
    """
    # Конвертируем NumPy (BGR) в PIL, затем снова в тензор — как делает RawDataset
    # или напрямую в тензор, повторяя логику из AlignCollate и т.д.
    try:
        pil_img = Image.fromarray(cv2.cvtColor(cropped_image_np, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.error(f"Error converting NumPy to PIL: {e}")
        return "", 0.0

    # Теперь имитируем логику AlignCollate (упрощённая версия):
    # 1) Resize PIL image до (opt_deep.imgW, opt_deep.imgH)
    # 2) Преобразуем в тензор PyTorch и нормируем
    pil_img = pil_img.convert('L') if not opt_deep.rgb else pil_img.convert('RGB')
    pil_img = pil_img.resize((opt_deep.imgW, opt_deep.imgH), Image.BICUBIC)

    img_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(pil_img.tobytes()))
    img_tensor = img_tensor.view(opt_deep.imgH, opt_deep.imgW, -1).permute(2, 0, 1).float()
    img_tensor.sub_(127.5).div_(127.5)  # Normalize to [-1, 1]

    img_tensor = img_tensor.unsqueeze(0).to(device)  # batch_size=1

    # Подготавливаем «пустую» последовательность для модели
    length_for_pred = torch.IntTensor([opt_deep.batch_max_length]).to(device)
    text_for_pred = torch.LongTensor(1, opt_deep.batch_max_length + 1).fill_(0).to(device)

    # Прогоняем через модель
    with torch.no_grad():
        preds = model_deep(img_tensor, text_for_pred, is_train=False)  # Attn
        # Если была бы CTC:
        # preds = model_deep(img_tensor, text_for_pred)

    # Декодируем предсказания
    _, preds_index = preds.max(2)
    preds_str = converter_deep.decode(preds_index, length_for_pred)

    # Если Attn, обрезаем по '[s]'
    pred = preds_str[0]
    if 'Attn' in opt_deep.Prediction:
        eos_idx = pred.find('[s]')
        if eos_idx != -1:
            pred = pred[:eos_idx]

    # Считаем confidence
    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)
    pred_max_prob = preds_max_prob[0]
    if 'Attn' in opt_deep.Prediction:
        if eos_idx != -1:
            pred_max_prob = pred_max_prob[:eos_idx]

    if pred_max_prob.numel() > 0:
        confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
    else:
        confidence_score = 0.0

    return pred, confidence_score

# =============================================================================
# 4. Функции для SFTP и загрузки изображений (без изменений)
# =============================================================================
def connect_sftp(host, port, username, password):
    try:
        transport = paramiko.Transport((host, port))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        logger.info("Connected to SFTP server.")
        return sftp, transport
    except Exception as e:
        logger.error(f"Failed to connect to SFTP server: {e}")
        return None, None

def download_images(sftp, remote_folder, local_folder):
    try:
        sftp.chdir(remote_folder)
        logger.info(f"Changed to directory: {remote_folder}")
    except Exception as e:
        logger.error(f"Failed to change directory {remote_folder}: {e}")
        return

    try:
        file_list = sftp.listdir()
        logger.info(f"Found {len(file_list)} files in folder {remote_folder}.")
    except Exception as e:
        logger.error(f"Failed to list files in folder {remote_folder}: {e}")
        return

    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    image_files = [f for f in file_list if f.lower().endswith(image_extensions)]

    logger.info(f"Found {len(image_files)} images to download.")

    downloaded_count = 0
    for filename in image_files:
        remote_file_path = os.path.join(remote_folder, filename)
        local_file_path = os.path.join(local_folder, filename)

        try:
            logger.info(f"Downloading file: {filename}")
            sftp.get(remote_file_path, local_file_path)
            logger.info(f"File downloaded successfully: {local_file_path}")
            downloaded_count += 1
        except Exception as e:
            logger.error(f"Failed to download file {filename}: {e}")
            continue

def safe_imread(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

# =============================================================================
# 5. Задачи Celery
# =============================================================================
@app.task
def download_images_task():
    #previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    remote_image_folder = os.path.join(REMOTE_BASE_FOLDER, previous_day, 'pic_001/')
    local_image_folder = os.path.join(LOCAL_BASE_FOLDER, 'images', previous_day)

    os.makedirs(local_image_folder, exist_ok=True)
    logger.info(f"Created local folder: {local_image_folder}")

    sftp, transport = connect_sftp(SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD)
    if sftp:
        download_images(sftp, remote_image_folder, local_image_folder)
        sftp.close()
        transport.close()
        logger.info("All images downloaded and connection closed.")

        # После загрузки — запускаем обработку
        logger.info(f"Enqueuing process_downloaded_images task for folder: {local_image_folder}")
        process_downloaded_images.delay(local_image_folder)


@app.task
def process_downloaded_images(image_folder):
    """Function to process downloaded images using YOLO and Deep Text Recognition."""
    logger.info(f"Starting to process images from folder: {image_folder}")
    previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Создаём выходные директории
    output_dir = os.path.join(LOCAL_BASE_FOLDER, 'results', previous_day)
    cropped_dir = os.path.join(output_dir, 'cropped')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)
    logger.info(f"Created results folder: {output_dir} and cropped images folder: {cropped_dir}")

    # Инициализируем YOLO-модель
    model_yolo = YOLO('/mnt/ks/Works/railcars/railcars_new/railcars_automatization/yolo_model.pt')
    # Инициализируем Deep Text Recognition (один раз на весь процесс)
    model_deep, converter_deep, opt_deep = init_deep_text_model()

    # Дополнительный отступ для боксов (padding)
    padding = 3

    # Обрабатываем каждый файл
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)

        if os.path.isfile(file_path):
            logger.info(f"Performing inference for {filename}...")
            image_cv2 = safe_imread(file_path)

            if image_cv2 is None:
                logger.error(f"Failed to load image: {filename}")
                continue

            # YOLO-inference
            results = model_yolo.predict(
                source=image_cv2,
                save=False,
                imgsz=640,
                conf=0.25
            )
            logger.info(f"Inference completed for {filename}.")

            # Создадим txt-файл для записи результатов OCR
            text_filename = f"{os.path.splitext(filename)[0]}.txt"
            text_path = os.path.join(output_dir, text_filename)

            with open(text_path, 'w', encoding='utf-8') as text_file:
                # Перебираем детектированные объекты
                for result in results:
                    boxes = result.boxes
                    if boxes:
                        logger.info(f"Detected {len(boxes)} objects in {filename}.")
                        for idx, box in enumerate(boxes):
                            xyxy = box.xyxy[0]
                            x_min, y_min, x_max, y_max = xyxy.tolist()

                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())

                            logger.info(
                                f"Class: {class_id}, Confidence: {confidence:.2f}, "
                                f"Box: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]"
                            )

                            # Увеличиваем bbox на несколько пикселей (padding)
                            x_min_int = int(max(x_min - padding, 0))
                            y_min_int = int(max(y_min - padding, 0))
                            x_max_int = int(min(x_max + padding, image_cv2.shape[1]))
                            y_max_int = int(min(y_max + padding, image_cv2.shape[0]))

                            cropped_image_np = image_cv2[y_min_int:y_max_int, x_min_int:x_max_int]

                            # Сохраняем вырезанный фрагмент
                            try:
                                cropped_filename = f"{os.path.splitext(filename)[0]}_crop_{idx}.jpg"
                                cropped_filepath = os.path.join(cropped_dir, cropped_filename)
                                cv2.imwrite(cropped_filepath, cropped_image_np)
                                logger.info(f"Cropped image saved: {cropped_filepath}")
                            except Exception as e:
                                logger.error(f"Error saving cropped image: {e}")
                                continue

                            # =========================
                            # Запускаем Deep Text OCR
                            # =========================
                            try:
                                ocr_text, conf_score = infer_text_deep(
                                    cropped_image_np,
                                    model_deep,
                                    converter_deep,
                                    opt_deep
                                )
                                # Запись результата
                                text_file.write(f"{ocr_text}\n")
                                logger.info(
                                    f"OCR result for class {class_id}: '{ocr_text}' "
                                    f"(confidence={conf_score:.4f})"
                                )
                            except Exception as e:
                                logger.error(f"OCR error or file write error: {e}")

            # Clean up memory
            del results
            gc.collect()
        else:
            logger.info(f"Skipped non-file: {filename}")

    logger.info("Processing complete. Results saved in the 'results' folder.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Стартуем задачу Celery (скачивание + обработка)
    download_images_task.delay()

