from celery.schedules import crontab

broker_url = 'redis://localhost:6379/0'

beat_schedule = {
    'download-every-morning': {
        'task': 'railcars_automatization.tasks.download_images_task',
        'schedule': crontab(hour=18, minute=20),  # Каждый день в 03:00
    },
#    'process_images_ocr_every_day': {
#        'task': 'celery.tasks.process_images_ocr_task',
#        'schedule': crontab(minute=50, hour=17),
#    },
#    'process_ocr_results_every_day': {
#        'task': 'celery.tasks.process_ocr_results_task',
#        'schedule': crontab(minute=00, hour=18),
#    },
}

timezone = 'Asia/Bishkek'
