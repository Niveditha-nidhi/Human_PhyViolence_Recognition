
from flask import Flask, request, render_template, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import smtplib





import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import cv2

UPLOAD_FOLDER = 'static/uploads'
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/start_processing', methods=['POST'])
def start_processing():
    data = request.get_json()
    filename = data['filename']
    global should_process_video
    should_process_video = True
    return jsonify({'message': 'Processing started'})


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        # Ensure the directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # Return the filename as a plain text response
        return filename


@app.route('/preview/<filename>')
def preview(filename):
    return render_template('preview.html', filename=filename)


@app.route('/video_feed/<filename>')
def video_feed(filename):
    global should_process_video
    if not should_process_video:
        return '', 204
    return Response(generate_frames('static/uploads/' + filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/processed_video_feed/<filename>')
def processed_video_feed(filename):
    return Response(generate_frames('static/uploads/' + filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




def generate_frames(video_path):
    model = load_model('Mobilenetv2model.h5', compile=False)

    image_height, image_width = 64, 64  # 128,128
    sequence_length = 16
    class_list = ["NonViolence", "Violence"]

    video_reader = cv2.VideoCapture(video_path)
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    print(f'The video has {fps} frames per second.')

    frames_queue = deque(maxlen=sequence_length)

    predicted_class_name = ''
    

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        resized_frame = cv2.resize(frame, (image_height, image_width))

        normalized_frame = resized_frame / 255

        frames_queue.append(normalized_frame)

        if len(frames_queue) == sequence_length:
            # input_data = np.expand_dims(frames_queue, axis=0)
            # predicted_labels_probabilities = model.predict([input_data, input_data])[0]  # provide the input twice
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]

            predicted_label = np.argmax(predicted_labels_probabilities)

            predicted_class_name = class_list[predicted_label]
            predicted_confidence = predicted_labels_probabilities[predicted_label]

        text = f'{predicted_class_name}'

        # Calculate the text size based on the video's height
        text_size = frame.shape[0] / 4  # Adjust the denominator to get the desired text size

        if predicted_class_name == "Violence":
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_size / 100, (0, 0, 255), 2)
            
        else:
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_size / 100, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_reader.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)