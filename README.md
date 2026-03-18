# AI Face Attendance System

An automated attendance system using facial recognition and blink detection to perform secure check-in and check-out via webcam.

---

## 🚀 Features

* Real-time face detection and recognition
* Blink detection for anti-spoofing (prevent fake check-in)
* Automatic check-in/check-out logging
* GUI-based interface (Tkinter)
* User registration system
* Dataset creation (video → images)
* Model training using KNN
* Manual attendance option

---

## 🧠 System Workflow

1. Register user (capture face data)
2. Convert video into image dataset
3. Train recognition model
4. Run system for real-time attendance
5. Blink detection confirms identity before logging

---

## 📁 Project Structure

```
ai-face-attendance/
 ├── main.py
 ├── register.py
 ├── classifier/
 │    └── trained_knn_model.clf
 ├── dataset/
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
* Tkinter
* NumPy, SciPy

---

## 📦 Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Required Model Files (IMPORTANT)

This project requires pre-trained facial landmark models (not included due to large size):

* `shape_predictor_68_face_landmarks.dat`
* `shape_predictor_5_face_landmarks.dat`

### Download:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

### Setup:

1. Download both files
2. Extract `.bz2` files
3. Place them in the project root folder:

```
ai-face-attendance/
 ├── shape_predictor_68_face_landmarks.dat
 ├── shape_predictor_5_face_landmarks.dat
```

---

## ▶️ Usage

### 1. Register a new user

```bash
python register.py
```

### 2. Run attendance system

```bash
python main.py
```

---

## 📌 Notes

* Webcam is required
* Good lighting improves accuracy
* Attendance is saved in `check.csv`
* Model is saved in `classifier/trained_knn_model.clf`

---

## 📥 Executable Version

If available, you can run the `.exe` version without installing Python.

---

## 🎯 Future Improvements

* Database integration (SQLite/MySQL)
* Web-based interface
* Multi-user management
* Cloud deployment
* Performance optimization

---

## 👨‍💻 Author

Computer vision project using AI, real-time processing, and GUI integration.
