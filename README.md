🐺 Witcher Hand Gesture Recognition

Cast magical Witcher signs (Aard, Quen, Igni, Yrden, Axii) in real-time using your bare hands!

📌 About The Project

This project is an interactive computer vision application that brings the Witcher universe into the real world. By leveraging MediaPipe for hand landmark detection and Scikit-Learn (k-NN) for machine learning classification, the system can recognize different hand gestures corresponding to Witcher signs in real-time through a webcam feed.

🛠️ Technologies Used

Python

OpenCV

Google MediaPipe Tasks API

Scikit-Learn

⚙️ How to Run Locally

1. Clone the Repository

```

git clone https://github.com/Wnmy42/Witcher-Hand-Gesture-Signs.git
cd Witcher-Hand-Gesture-Signs

```

2. Install Dependencies

Make sure you have Python installed, then run:

```

pip install opencv-python mediapipe scikit-learn pandas numpy

```

3. Collect Data for Signs
   
To train the model with your own hand gestures:

```

python collect_data.py

```

(Press 's' to capture samples for each sign category).

4. Train the Model

Run the training script to process the dataset and generate the model file:

```

python train_model.py

```

5. Run Live Detection

Launch the real-time application:

```

python run_witcher.py

```

(Press 'q' to exit the camera window).

🔮 Supported Witcher Signs

Aard 💨 - Telekinetic thrust

Quen 🛡️ - Protective shield

Igni 🔥 - Flame stream

Yrden 🌀 - Magical trap

Axii 💫 - Mind control

🤝 Contributing

Contributions, issues, and feature requests are welcome!
