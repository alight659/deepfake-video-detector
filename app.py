import os

#from sqlite import SQLite
from flask import Flask, flash, render_template, request, redirect, session, Response, url_for, send_from_directory
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from predict import predictor


app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Error Handling 404
@app.errorhandler(404)
def error(code):
    return render_template("error.html", code=code), 404


# Error Handling 500
@app.errorhandler(500)
def error(code):
    return render_template("error.html", code=code), 500


# Error Handling 400
@app.errorhandler(400)
def error(code):
    return render_template("error.html", code=code), 400


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))
    return redirect("/predict")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/predict", methods=["GET", "POST"])
def model_predict():
    if 'video' not in request.files:
        return redirect(request.url)
    video = request.files['video']
    if video and allowed_file(video.filename):
        filename = video.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)

        s = predictor(filepath)
        
        if s[0] > 0:
            return render_template("predict.html", accuracy=round(s[0] * 100), status=s[1], path=url_for('uploaded_file', filename=filename))
        else:
            return render_template("predict.html", status=s[1], path=url_for('uploaded_file', filename=filename))

    return render_template("predict.html")


if __name__ == "__main__":
    app.run()

    
