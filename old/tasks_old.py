
import multiprocessing

# Установите метод старта на 'spawn'
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from celery import Celery
import os
import paramiko
import logging
from datetime import datetime, timedelta
import easyocr
import cv2

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

REMOTE_BASE_FOLDER = '/rmdocfolder/image/DH-IPC-HFW3441DGP-AS-4G-NL668EAU-B-0360B/'
LOCAL_BASE_FOLDER = '/mnt/ks/Works/railcars/railcars_new'



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

    downloaded_count = 0  # Инициализация переменной здесь
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
            
def ocr_number(image, reader):
    cropped_image = image[90:450, :]
    #gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    #inverted_image = cv2.bitwise_not(gray_image)
    results = reader.readtext(cropped_image, detail=0, allowlist='0123456789', min_size=60)
    return results


def safe_imread(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None
        
@app.task
def download_images_task():
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

        # Обработка загруженных изображений сразу после загрузки
        process_downloaded_images.apply_async(args=[local_image_folder])


@app.task
def process_downloaded_images(image_folder):
    """Function to process downloaded images using OCR."""
    logger.info(f"Starting to process images from folder: {image_folder}")
    reader = easyocr.Reader(['en'])  # Initialize the reader inside the task
    previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Create folder 'results/<date>' if it doesn't exist
    text_folder = os.path.join(LOCAL_BASE_FOLDER, 'results', previous_day)
    os.makedirs(text_folder, exist_ok=True)
    logger.info(f"Created results folder: {text_folder}")

    # Process each image in the 'images' folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(image_folder, filename)
            image = safe_imread(image_path)

            if image is None:
                logger.error(f"Error loading image: {filename}. Skipping.")
                continue

            try:
                logger.info(f"Processing image: {filename} for OCR")
                results = ocr_number(image, reader)
                logger.info(f"OCR results for {filename}: {results}")

                if not results:
                    logger.warning(f"No text found in image: {filename}. Skipping text file creation.")
                    continue

                text_filename = os.path.splitext(filename)[0] + '.txt'
                text_path = os.path.join(text_folder, text_filename)

                with open(text_path, 'w', encoding='utf-8') as text_file:
                    for t in results:
                        text_file.write(t + '\n')

                logger.info(f"Saved OCR results to {text_path}")

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")

    logger.info("Processing complete. Number-only text files saved in the 'results' folder.")
