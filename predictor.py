import cv2
import numpy as np
from gtts import gTTS
from IPython.display import Audio, display

def predict_gesture(model, img, label_map):
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = img.reshape(1,28,28,1)

    pred = model.predict(img)
    label = np.argmax(pred)
    gesture = label_map.get(label, "Unknown")
    print("Predicted Gesture:", gesture)

    speech = gTTS(text=gesture, lang='en', slow=False)
    speech.save("gesture.mp3")
    display(Audio("gesture.mp3", autoplay=True))
    return gesture
