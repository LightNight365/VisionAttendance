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
 ├── Final_ID2.py
 ├── registerGUIdone.py
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

### 1. Install dependencie
