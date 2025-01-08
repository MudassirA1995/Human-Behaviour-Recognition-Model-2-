import cv2
from fer import FER
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget, QProgressBar
from PyQt5.QtGui import QImage, QPixmap
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
        self.video_label.setAlignment(Qt.AlignCenter)
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
        self.timer.timeout.connect(self.update_frame)

        # Initialize FER detector and OpenCV face detector
        self.detector = FER(mtcnn=False)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def toggle_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.emotion_label.setText("Error: Camera not accessible")
                self.capture = None
                return
            self.timer.start(33)  # ~30 FPS (1000 ms / 30)
            self.control_button.setText("Stop Camera")
        else:
            self.timer.stop()
            self.capture.release()
            self.capture = None
            self.video_label.clear()
            self.control_button.setText("Start Camera")

    def update_frame(self):
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            self.emotion_label.setText("Error: Could not read frame")
            return

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # For each detected face, apply emotion detection
        for (x, y, w, h) in faces:
            # Draw a red rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Crop the face region and apply emotion detection
            face_region = frame[y:y + h, x:x + w]
            emotions = self.detector.top_emotion(face_region)
            
            if emotions:
                dominant_emotion, score = emotions
                self.emotion_label.setText(f"Emotion: {dominant_emotion.capitalize()} - {score*100:.2f}%")

                # Update the emotion progress bars
                self.update_progress_bars(emotions)

        # Convert frame to QImage for PyQt5 display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        qt_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Display the frame in the QLabel
        self.video_label.setPixmap(pixmap)
        self.video_label.setScaledContents(True)

    def update_progress_bars(self, emotions):
        dominant_emotion, score = emotions

        # Reset bars
        for emotion, bar in self.progress_bars.items():
            bar.setValue(0)

        # Set the bar for the dominant emotion
        if dominant_emotion.lower() in self.progress_bars:
            self.progress_bars[dominant_emotion.lower()].setValue(int(score * 100))

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



