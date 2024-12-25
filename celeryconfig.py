#celeryconfig.py
from celery import Celery
from celery.schedules import crontab

app = Celery('railcars_automatization')
app.config_from_object('railcars_automatization.celeryconfig')

broker_url = 'redis://localhost:6379/0'


beat_schedule = {
    'download-every-day': {
        'task': 'railcars_automatization.tasks.download_images_task',
        'schedule': crontab(hour=6, minute=00),
    },
    'import-data-to-db-every-day': {
        'task': 'railcars_automatization.import_db.import_data_to_db',
        'schedule': crontab(hour=6, minute=20),
    },
}

timezone = 'Asia/Bishkek'




