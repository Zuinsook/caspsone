
from flask import Flask, render_template, request
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template

import ssl
import requests, crawling
from bs4 import BeautifulSoup

import os
import sys
import numpy as np


from keras.models import load_model
from werkzeug.utils import secure_filename
from utils.utils import process_predictions
from feature_extraction.makefeat import featureextract



app = Flask(__name__) #,static_url_path='/static')

#
# @app.route('/')
# def home():  # put application's code here
#     return render_template('index.html')
#
#
# @app.route('/stt.html', methods=['GET', 'POST'])
# def stt():  # put application's code here
#     return render_template('stt.html')
#
#
# @app.route('/ml.html', methods=['GET', 'POST'])
# def ml():  # put application's code here
#     return render_template('ml.html')
#
#
# @app.route('/news', methods=['GET','POST'])
# def news():
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
#     data = requests.get('https://www.fnnews.com/search?search_txt=청각장애인&page=0', headers=headers)
#     soup = BeautifulSoup(data.text, 'html.parser')
#     news = soup.select('#root > div.contents > div > div.wrap_left > div.wrap_artlist.mt20 > div.inner_artlist > ul > li')
#     news_dict={}
#     for i in range(9):
#         for new in news:
#             a_tag = new.select_one('a')
#             if a_tag is not None:
#                 url='https://www.fnnews.com'
#                 url+=new.select_one('a:nth-child(1)')['href']
#                 image = new.select_one('a:nth-child(1) > span > img')['src']
#                 title = a_tag.select_one('strong').text.strip()
#                 desc = new.select_one('a:nth-child(2) > p').text.strip()
#                 date = new.select_one('em').text
#                 news_dict[str(url)]={
#                     'img':image,
#                     'title':title,
#                     'desc':desc,
#                     'date':date
#                 }
#                 print('크롤링 완료')
#
#     return render_template('news.html',news_dict=news_dict)
#
import os
import sys
import numpy as np


from keras.models import load_model
from werkzeug.utils import secure_filename
from utils.utils import process_predictions
from feature_extraction.feature_extractor import FeatureExtractor
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'opus'}


# Limit content size
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(filepath):
    feature_extractor = featureextract('config_files/feature_extraction.json')
    return feature_extractor.extract_features(filepath)


# Upload files function
@app.route('/', methods=['GET', 'POST']) #'/upload_file'
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file2']  #소리 받은거
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
            file.save(filename)
            return redirect(url_for('classify_and_show_results', filename=filename))
    return render_template("index3.html")#index3.html")

#record 파일 to result.html
# @app.route('/', methods=['GET', 'POST']) #'/upload_file'
# def upload_file():
#     if request.method


# Classify and show results
@app.route('/results', methods=['GET'])
def classify_and_show_results():
    filename = request.args['filename']
    # Compute audio signal features
    features = extract_features(filename)
    features = np.expand_dims(features, 0)
    # Load model and perform inference
    model = load_model('models/best_model.hdf5')
    predictions = model.predict(features)[0]
    # Process predictions and render results
    predictions_probability, prediction_classes = process_predictions(predictions,'config_files/classes.json')

    predictions_to_render = {prediction_classes[i]:"{}%".format(round(predictions_probability[i]*100, 3)) for i in range(3)}
    # Delete uploaded file
    os.remove(filename)
    # Render results
    return render_template("results.html", filename=filename, predictions_to_render=predictions_to_render)



# start recording
#
# @app.route('/record', methods=['GET'])
# def classify_and_show_results():
#
#     return render_template("record.html")

import librosa
import librosa.display
# import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import glob
from IPython import get_ipython
def audio_preprocessing(data,index):


def record(test):
    import pyaudio
    import wave

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    # WAVE_OUTPUT_FILENAME = "./audio_data/output_"+str(j)+".wav"
    WAVE_OUTPUT_FILENAME = "./audio_data/output_" + test + ".wav"
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME






if __name__ == '__main__':
    # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    # ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem', password='louie')
    app.run(host="0.0.0.0", port=5000) #1, ssl_context=ssl_context)

