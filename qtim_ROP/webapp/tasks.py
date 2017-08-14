from celery import Celery
from flask import Flask, flash, request, redirect, url_for, render_template, session, send_from_directory
from flask.json import jsonify
from werkzeug.utils import secure_filename
import random
import time
import os

def make_celery(app):
    celery = Celery('tasks') #, backend=app.config['CELERY_RESULT_BACKEND'],
                    #broker=app.config['CELERY_BROKER_URL'])
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
    UPLOAD_FOLDER='uploads/raw'
)

celery = make_celery(app)

@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of long task
    if 'file' not in request.files:
        print "No file found"
        return render_template('index.html')

    file = request.files['file']

    if file.filename == '':
        flash('No file selected!')
        return redirect(request.url)

    if file and valid_image(file.filename):
        filename = secure_filename(file.filename)
        print filename
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(out_path)
        
        task = run_inference.apply_async([filename])
        return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}


# @app.route('/classify_image/<filename>', methods=['GET', 'POST'])
# def classify_image(filename):

#     # Create instance of long task
#     image_path = request.args.get('filename')
#     print image_path
    

@celery.task(bind=True)
def run_inference(self, image_path):

    print "Long task running..."
    """Background task that runs a long function with progress reports."""
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = random.randint(10, 50)
    for i in range(total):
        print i
        if not message or random.random() < 0.25:
            message = '{0} {1} {2}...'.format(random.choice(verb),
                                              random.choice(adjective),
                                              random.choice(noun))
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': total,
                                'status': message})
        time.sleep(.5)
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42}

@app.route('/status/<task_id>')
def taskstatus(task_id):

    task = run_inference.AsyncResult(task_id)
    if task.state == 'PENDING':

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