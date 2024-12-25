from celery import Celery

app = Celery('celery')
app.config_from_object('celery.celeryconfig')

# Import tasks
from celery.tasks import download_images_task
