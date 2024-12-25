
import os
import gc
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
import paramiko
import logging
from datetime import datetime, timedelta

from celery import Celery
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

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

def ocr_number(image, reader):
    try:
        results = reader.readtext(image, detail=0, allowlist='0123456789',min_size=60,text_threshold = 0.6)
        return results
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return []

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

        # Process downloaded images after downloading
        logger.info(f"Enqueuing process_downloaded_images task for folder: {local_image_folder}")
        process_downloaded_images.delay(local_image_folder)

@app.task
def process_downloaded_images(image_folder):
    """Function to process downloaded images using YOLO and OCR."""
    logger.info(f"Starting to process images from folder: {image_folder}")
    #previous_day = datetime.now().strftime('%Y-%m-%d')
    previous_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    # Create output directories
    output_dir = os.path.join(LOCAL_BASE_FOLDER, 'results', previous_day)
    cropped_dir = os.path.join(output_dir, 'cropped')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)
    logger.info(f"Created results folder: {output_dir} and cropped images folder: {cropped_dir}")

    # Initialize OCR and YOLO model here
    reader = easyocr.Reader(['en'])
    model = YOLO('/mnt/ks/Works/railcars/railcars_new/test1/best.pt')

    # Increase bounding box size
    padding = 3  # Increase coordinates by 5 pixels

    # Process each image in the folder
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)

        if os.path.isfile(file_path):
            logger.info(f"Performing inference for {filename}...")
            image_cv2 = safe_imread(file_path)

            if image_cv2 is None:
                logger.error(f"Failed to load image: {filename}")
                continue

            # Perform inference
            results = model.predict(
                source=image_cv2,
                save=False,
                imgsz=640,
                conf=0.25
            )
            logger.info(f"Inference completed for {filename}.")

            # Create file to write OCR results
            text_filename = f"{os.path.splitext(filename)[0]}.txt"
            text_path = os.path.join(output_dir, text_filename)

            with open(text_path, 'w', encoding='utf-8') as text_file:
                # Process detection results
                for result in results:
                    boxes = result.boxes
                    if boxes:
                        logger.info(f"Detected {len(boxes)} objects in {filename}.")
                        for idx, box in enumerate(boxes):
                            xyxy = box.xyxy[0]
                            x_min, y_min, x_max, y_max = xyxy.tolist()

                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())

                            logger.info(f"Class: {class_id}, Confidence: {confidence:.2f}, Box: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")

                            # Increase bounding box size with padding
                            x_min_int = int(max(x_min - padding, 0))
                            y_min_int = int(max(y_min - padding, 0))
                            x_max_int = int(min(x_max + padding, image_cv2.shape[1]))
                            y_max_int = int(min(y_max + padding, image_cv2.shape[0]))

                            cropped_image_np = image_cv2[y_min_int:y_max_int, x_min_int:x_max_int]

                            # Save cropped image
                            try:
                                cropped_filename = f"{os.path.splitext(filename)[0]}_crop_{idx}.jpg"
                                cropped_filepath = os.path.join(cropped_dir, cropped_filename)
                                cv2.imwrite(cropped_filepath, cropped_image_np)
                                logger.info(f"Cropped image saved: {cropped_filepath}")
                            except Exception as e:
                                logger.error(f"Error saving cropped image: {e}")
                                continue

                            # Apply OCR
                            try:
                                ocr_results = ocr_number(cropped_image_np, reader)
                                ocr_text = ' '.join(ocr_results)

                                # Write OCR result
                                text_file.write(f"{ocr_text}\n")
                                logger.info(f"OCR result for class {class_id}: {ocr_text}")
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
    
    # Start the task
    download_images_task.delay()
