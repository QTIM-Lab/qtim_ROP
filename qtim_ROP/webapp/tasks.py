from celery import Celery
from flask import Flask, flash, request, redirect, url_for, render_template, session, send_from_directory
from flask.json import jsonify
from werkzeug.utils import secure_filename
import os
import time
import numpy as np

from qtim_ROP.__main__ import initialize
from qtim_ROP.deep_rop import preprocess_images, LABELS
from qtim_ROP.learning.retina_net import RetiNet, locate_config

def make_celery(app):

    celery = Celery('tasks')
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask

    return celery

def valid_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])
app.config.update(
    CELERY_BROKER_URL='amqp://localhost:6379',
    CELERY_RESULT_BACKEND='amqp://localhost:6379',
    UPLOAD_FOLDER='uploads/raw',
    OUTPUT_FOLDER='uploads/output',
    SEG_FOLDER='uploads/output/segmentation'
)

celery = make_celery(app)

# Initialize model
conf_dict, conf_file = initialize()
classifier_dir = conf_dict['classifier_directory']
model_config, rf_pkl = locate_config(classifier_dir)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/upload", methods=["POST"])
def upload():

    # Create instance of long task
    if 'file' not in request.files:
        print("No file found")
        return render_template('index.html')

    file = request.files['file']

    if file.filename == '':
        flash('No file selected!')
        return redirect(request.url)

    if file and valid_image(file.filename):
        filename = secure_filename(file.filename)
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(out_path)
        return render_template("uploaded.html", image_name=filename, seg_name=None)


@app.route('/upload/<filename>')
def send_original(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload/<seg_name>')
def send_segmentation(seg_name):
    print(seg_name)
    return send_from_directory(app.config['SEG_FOLDER'], seg_name)


@app.route('/longtask', methods=['POST'])
def longtask():

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.form['filename'])
    task = long_task.apply_async([file_path])
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}

@celery.task(bind=True)
def long_task(self, filename):

    # Initializes the classifer
    self.update_state(state='PROGRESS', meta={'current': 0, 'total': 3, 'status': "Initializing..."})
    time.sleep(2)
    classifier = RetiNet(model_config)

    # Does the preprocessing
    self.update_state(state='PROGRESS', meta={'current': 1, 'total': 3, 'status': "Preprocessing image..."})
    prep_img, prep_name = preprocess_images([filename], app.config['OUTPUT_FOLDER'],
                                            conf_dict, skip_segmentation=False, batch_size=100, fast=True)
    self.update_state(state='PROGRESS', meta={'current': 2, 'total': 3, 'status': "Doing classification..."})

    # Prediction
    y_preda = classifier.predict(prep_img)[0]
    arg_max = np.argmax(y_preda)
    prob = y_preda[arg_max]
    diagnosis = LABELS[arg_max]

    return {'current': 3, 'total': 3, 'status': 'Complete!', 'seg_name': prep_name,
            'result': '{} ({:.2f}%)'.format(diagnosis, prob * 100.)}

@app.route('/status/<task_id>')
def taskstatus(task_id):

    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
            response['seg_name'] = task.info['seg_name']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)