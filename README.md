# AI Face Attendance System

An automated attendance system using facial recognition and blink detection to perform secure check-in and check-out via webcam.

---

## 🚀 Features

* Real-time face detection and recognition
* Blink detection to prevent spoofing (anti-fake check)
* Automatic check-in/check-out logging
* User registration system with GUI
* Dataset creation (video → images)
* Model training using KNN
* Manual attendance option

---

## 🧠 How It Works

1. Register a new user (capture face data via webcam)
2. Convert recorded video into image dataset
3. Train facial recognition model (KNN)
4. Run the main system for real-time recognition
5. System logs attendance automatically when a face is recognized and blink is detected

---

## 📁 Project Structure

```
ai-face-attendance/
 ├── Final_ID2.py
 ├── registerGUIdone.py
 ├── classifier/
 │    └── trained_knn_model.clf
 ├── dataset/
 ├── shape_predictor_68_face_landmarks.dat
 ├── shape_predictor_5_face_landmarks.dat
 ├── circle.png
 ├── check.csv
 ├── requirements.txt
 └── README.md
```

---

## ⚙️ Technologies Used

* Python
* OpenCV
* face_recognition (dlib)
* scikit-learn (KNN)
* Tkinter (GUI)
* NumPy, SciPy

---

## 📦 Installation

1. Clone or download the project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Register a new user

```bash
python register.py
```

### 2. Run the attendance system

```bash
python main.py
```

---

## 📌 Notes

* A webcam is required
* Ensure good lighting for better accuracy
* Model file is stored in `classifier/trained_knn_model.clf`
* Attendance is saved in `check.csv`

---

## 📥 Executable Version (Optional)

If you do not want to run the code manually, you can use the compiled `.exe` version (if provided in Releases or project files).

---

## 🎯 Future Improvements

* Database integration (SQLite/MySQL)
* Web-based interface
* Multi-user management system
* Improved face recognition accuracy
* Cloud deployment

---

## 👨‍💻 Author

Developed as a computer vision project using AI and real-time image processing.
