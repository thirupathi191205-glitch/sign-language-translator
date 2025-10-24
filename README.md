Sign Language Translator using Deep Learning

Overview
This project is a **Sign Language Translator** built using **Deep Learning (TensorFlow & CNN)**.  
It recognizes hand gestures representing **letters (A–F)** from the **Sign Language MNIST dataset** and translates them into **text and speech** using **Google Text-to-Speech (gTTS)**.

The model processes input gestures, predicts the corresponding letter or word (like "Hello", "Thank You", etc.), and converts it to **audio output**, bridging the communication gap between sign language users and others.

---

 Project Structure
SignLanguageTranslator/
│
├── README.md
├── requirements.txt
├── gesture_model.keras
│
├── data/
│ ├── sign_mnist_train.csv
│ └── sign_mnist_test.csv
│
├── src/
│ ├── data_loader.py
│ ├── model_builder.py
│ ├── trainer.py
│ ├── predictor.py
│ └── webcam_capture.py
│
├── utils/
│ └── label_map.py
│
└── main.py

yaml
Copy code

---

 Requirements

Ensure you have **Python 3.8+** installed.  
Install dependencies using:
```bash
pip install -r requirements.txt
Dependencies
TensorFlow

NumPy

OpenCV

Pandas

gTTS

Mediapipe

KaggleHub

 Dataset
Dataset: Sign Language MNIST (Kaggle)
This dataset contains 28x28 grayscale images representing American Sign Language (A–Y, except J and Z).

Place the downloaded files inside the data/ folder:

kotlin
Copy code
data/
├── sign_mnist_train.csv
└── sign_mnist_test.csv
 How to Run the Project
Step 1: Clone the repository
bash
Copy code
git clone https://github.com/yourusername/SignLanguageTranslator.git
cd SignLanguageTranslator
Step 2: Install dependencies
bash
Copy code
pip install -r requirements.txt
Step 3: Train the model
The model is automatically trained on the Sign Language MNIST dataset (A–F).

bash
Copy code
python main.py
During training:

The model builds a CNN using TensorFlow

Trains for 5 epochs

Saves the model as gesture_model.keras

 Prediction and Translation
After training, the webcam interface opens (works best in Google Colab).
You can show a sign (A–F) to the camera for prediction.
The model will:

Recognize the gesture

Display the predicted label (e.g., “Hello”, “Thank You”)

Speak the word aloud using gTTS

 Label Mapping
Defined in utils/label_map.py:

Label	Word
0	Hello
1	Thank You
2	Welcome
3	Good Morning
4	Good Night
5	Eat

 Model Architecture
Layer	Type	Activation	Output Shape
1	Conv2D (32 filters)	ReLU	28x28x32
2	MaxPooling2D	-	14x14x32
3	Conv2D (64 filters)	ReLU	12x12x64
4	MaxPooling2D	-	6x6x64
5	Flatten	-	2304
6	Dense (128)	ReLU	128
7	Dense (6)	Softmax	6

 Key Features
1. Real-time gesture recognition
2. Speech output using gTTS
3. Uses CNN for high accuracy
4. Works with Google Colab or local Python
5. Modular code structure (easy to expand)

 Future Enhancements
Add support for full A–Z gesture recognition

Improve accuracy using Mediapipe-based hand landmarks

Integrate real-time webcam recognition outside Colab

Build a user-friendly desktop or web interface

 Author
Developed by: Your Name
College: Panimalar Engineering College – BE CSE
Year: 5th Semester
Subject: Deep Learning Project

 License
This project is licensed under the MIT License – you are free to use and modify it.

 Bridging communication barriers with the power of AI and Deep Learning! 

yaml
Copy code

---

Would you like me to include **Colab setup instructions** (with GPU settings and notebook execution commands) inside the README too?  
That would make it ready to run directly in Google Colab.





