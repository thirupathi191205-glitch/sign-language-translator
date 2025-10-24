from IPython.display import display, Javascript
from google.colab import output
import cv2
import numpy as np
from base64 import b64decode
from src.predictor import predict_gesture
from utils.label_map import label_map

gesture_prediction = ""

def capture_image(model):
    display(Javascript('''
        async function takePhoto() {
            const div = document.createElement('div');
            const video = document.createElement('video');
            div.appendChild(video);
            document.body.appendChild(div);
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            const dataUrl = canvas.toDataURL('image/jpeg', 1.0);
            stream.getTracks().forEach(track => track.stop());
            div.remove();
            google.colab.kernel.invokeFunction('notebook.capture', [dataUrl], {});
        }
        takePhoto();
    '''))

    def decode_image(dataUrl):
        header, encoded = dataUrl.split(",", 1)
        data = b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        predict_gesture(model, img, label_map)

    output.register_callback('notebook.capture', decode_image)
