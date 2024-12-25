#automate_railcar_tasks.py
from celery import Celery

app = Celery('railcars_automatization')
app.config_from_object('railcars_automatization.celeryconfig')

# Import tasks
from railcars_automatization.tasks import download_images_task
from railcars_automatization.import_db import import_data_to_db
