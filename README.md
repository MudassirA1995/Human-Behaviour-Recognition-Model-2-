# Human Emotion Recognition Model 2 

## Introduction 

The Emotion Recognition App is a PyQt5-based GUI application designed to analyze live video feed from a webcam and detect the emotions displayed on a person's face. The application utilizes the FER (Facial Expression Recognition) library to analyze facial expressions and display the detected emotions in real-time.

## Usecases 

1) **Mental Health Monitoring**: Assists in identifying emotional states to provide real-time feedback for mental health applications.
2) **Customer Service**: Can be used to gauge customer reactions and emotions during interactions.
3) **Educational Tools**: Provides emotion tracking for interactive learning environments or presentations.
4) **Human-Computer Interaction**: Enhances AI systems by adding emotional intelligence for personalized responses.

## Source Code 

```python

import cv2
from fer import FER
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QPushButton, QWidget, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import QTimer, Qt

class EmotionRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Setup the UI
        self.setWindowTitle("Emotion Recognition")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: black;")

        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Video feed
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        # Detected emotion
        self.emotion_label = QLabel("Emotion: Detecting...", self)
        self.emotion_label.setStyleSheet("font-size: 20px; color: #00FFFF;")
        self.layout.addWidget(self.emotion_label)

        # Emotion probability bars
        self.progress_bars = {}
        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        for emotion in emotions:
            bar = QProgressBar(self)
            bar.setStyleSheet(
                "QProgressBar { background-color: black; border: 2px solid #00FFFF; text-align: center; color: #00FFFF; } "
                "QProgressBar::chunk { background-color: #00FFFF; }"
            )
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setTextVisible(False)
            self.layout.addWidget(bar)
            self.progress_bars[emotion] = bar

        # Start/Stop button
        self.control_button = QPushButton("Start Camera", self)
        self.control_button.setStyleSheet("color: #00FFFF; background-color: black; border: 2px solid #00FFFF;")
        self.control_button.clicked.connect(self.toggle_camera)
        self.layout.addWidget(self.control_button)

        # Initialize camera and timer
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.detect_emotion)

        # Initialize FER detector
        self.detector = FER(mtcnn=True)

    def toggle_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.emotion_label.setText("Error: Camera not accessible")
                return
            self.timer.start(50)
            self.control_button.setText("Stop Camera")
        else:
            self.timer.stop()
            self.capture.release()
            self.capture = None
            self.video_label.clear()
            self.control_button.setText("Start Camera")

    def detect_emotion(self):
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Detect emotions
        results = self.detector.detect_emotions(frame)
        if results:
            emotions = results[0]["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            self.emotion_label.setText(f"Emotion: {dominant_emotion}")

            # Update progress bars
            for emotion, bar in self.progress_bars.items():
                bar.setValue(int(emotions.get(emotion, 0) * 100))

            # Draw rectangle around the face
            box = results[0].get("box", None)
            if box:
                x, y, w, h = box
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert frame to QImage for PyQt5 display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()
        event.accept()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = EmotionRecognitionApp()
    window.show()
    sys.exit(app.exec_())

```

## Code Explanation 

Class Definition:

The EmotionRecognitionApp class inherits from QWidget and manages the GUI layout, video feed, and emotion detection logic.

UI Setup:

The user interface is created with PyQt5 components like QVBoxLayout, QLabel, QPushButton, and QProgressBar. A black and neon blue color theme is applied for an attractive look.

Camera Control:

The toggle_camera method initializes the webcam using OpenCV and toggles the video feed on or off. It updates the button text accordingly.

Emotion Detection:

The detect_emotion method reads frames from the camera, flips them for a mirror effect, and passes them to the FER detector for analysis. Detected emotions and their probabilities are displayed in progress bars, and the dominant emotion is highlighted.

Face Highlighting:

A rectangle is drawn around the detected face for visual feedback, enhancing the user experience.

Cleanup:

The closeEvent method ensures the webcam is released properly when the application is closed.

## Libraries Used 

FER (Facial Expression Recognition):

Detects emotions such as angry, happy, sad, etc., from facial expressions in the video frames. Uses MTCNN for better face detection accuracy.

OpenCV:

Captures video from the webcam and processes the frames for display and analysis.

PyQt5:

Creates the graphical user interface, handles layouts, and manages user interactions.

QtCore, QtGui:

Provides additional classes for styling, image processing, and timer functionality.


## Application View 

![Screenshot (557)](https://github.com/user-attachments/assets/72e7ef06-1f89-45b9-8370-571600b20c96)













