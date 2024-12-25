# tasks.py
from celery import Celery
import os
import paramiko
import logging
from datetime import datetime, timedelta

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

# Function to connect to SFTP
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

# Function to download images from SFTP
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

    for filename in image_files:
        remote_file_path = os.path.join(remote_folder, filename)
        local_file_path = os.path.join(local_folder, filename)

        try:
            logger.info(f"Downloading file: {filename}")
            sftp.get(remote_file_path, local_file_path)
            logger.info(f"File downloaded successfully: {local_file_path}")
        except Exception as e:
            logger.error(f"Failed to download file {filename}: {e}")
            continue

# Celery task to download images
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
