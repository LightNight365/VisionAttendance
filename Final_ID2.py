#Thư viện sử dụng
import tkinter as tk                #tạo giao diện
from tkinter import Label
import cv2                          #Thư viện xử lý ảnh
from PIL import Image, ImageTk      #Quản lý hình ảnh trên GUI
import threading                    #Chạy song song việc
import dlib                         #phân tích đặc điểm gương mặt
import face_recognition             #nhận diện gương mặt
import imutils                      
from imutils import face_utils      #sử dụng các đặc điểm gương mặt
from scipy.spatial import distance as dist  #tính khoảng cách
import time                         #thời gian
import pickle                       #Lưu dữ liệu cho model
import csv                          #tạo file csv
import os                           #Quản lý đường dẫn, địa chỉ
import sys                          #thoát chương trình
from datetime import datetime

#buộc tắt chương trình
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# ĐỊA CHỈ CÁC FILE CẦN THIẾT
base_path = os.path.dirname(os.path.abspath(__file__))
shape_predictor_path = os.path.join(base_path, 'shape_predictor_68_face_landmarks.dat')
shape_predictor_5_path = os.path.join(base_path, "shape_predictor_5_face_landmarks.dat")
#knn_model_path = os.path.join(base_path,"registerGUIdone", "classifier","trained_knn_model.clf")
knn_model_path = os.path.join(base_path, "classifier","trained_knn_model.clf")
file_path = os.path.join(base_path, "check.csv")

class WebcamGUI:
    def __init__(self, window, window_title, video_source=0):

        #tạo giao diện
        self.window = window
        self.window.title(window_title)
        original_image = Image.open("circle.png")
        self.loadimage = ImageTk.PhotoImage(original_image.resize((72, 72)))

        self.window.attributes('-fullscreen', True)  
        self.window.bind('<Escape>', self.exit_fullscreen)  
        self.window.bind('q', self.close_windows)
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        self.prediction_label = Label(window, text="Predicted: ", font=('', 16), bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.ok_button = tk.Button(window, text="OK", bg='green', image=self.loadimage, compound='center', command=self.ok_pressed, bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.close_button = tk.Button(window, text="CLOSE", bg='yellow', image=self.loadimage, compound='center', command=self.close_windows, bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.back_button = tk.Button(window, text="BACK", bg='red', image=self.loadimage, compound='center', command=self.back_pressed, bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.reg_button = tk.Button(window, text="REGISTER", bg='orange', image=self.loadimage, compound='center', command=self.register_window, bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.manual_button = tk.Button(window, text="MANUAL", bg='blue', image=self.loadimage, compound='center', command=self.open_manual_window, bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.video_label = Label(window, bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.red_light = Label(window, bg='red', bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.green_light = Label(window, bg='green', bd=2, relief="solid", highlightbackground="black", highlightthickness=2)

        self.prediction_label.place(anchor='s', relx=0.5, rely=0.45, y=-180)
        self.ok_button.place(anchor='ne', relx=0.5, rely=0.5, x=-160, y=-160)
        self.close_button.place(anchor='nw', relx=0.5, rely=0.5, x=160, y=-160)
        self.back_button.place(anchor='se', relx=0.5, rely=0.5, x=-160, y=160)
        self.reg_button.place(anchor='sw', relx=0.5, rely=0.5, x=160, y=160)
        self.manual_button.place(anchor='center', relx=0.5, rely=0.5, y=200)
        self.video_label.place(anchor='center', relx=0.5, rely=0.5, width=300, height=300)
        self.red_light.place(anchor='n', relx=0.5, rely=0.5, x=-160, y=180, width=75, height=75)
        self.green_light.place(anchor='n', relx=0.5, rely=0.5, x=160, y=180, width=75, height=75)

        self.run_main_py_thread = None
        self.button_press_count = 0
        self.last_pred_name = None
        self.recognition_count = {}

        self.set_light('red')
        self.window.mainloop()

    def exit_fullscreen(self, event=None):
        self.window.attributes('-fullscreen', False)

    def set_light(self, color):
        if color == 'red':
            self.red_light.config(bg='red')
            self.green_light.config(bg='gray')
        elif color == 'green':
            self.red_light.config(bg='gray')
            self.green_light.config(bg='green')

    def close_windows(self):
        if self.vid.isOpened(): 
            self.vid.release()
        self.window.destroy()
        cv2.destroyAllWindows()

    def register_window(self, event=None):
        exe_path = os.path.join(os.path.dirname(__file__), "registerGUIdone", "registerGUIdone.exe")
        os.startfile(exe_path)

    def ok_pressed(self):
        if self.run_main_py_thread is None or not self.run_main_py_thread.is_alive():
            self.run_main_py_thread = threading.Thread(target=self.recognize_and_check_blinks)
            self.run_main_py_thread.start()

    def back_pressed(self):
        self.prediction_label.config(text="Predicted: ")
        self.last_pred_name = None  
        self.recognition_count = {}  


        self.delete_last_row()

        if self.run_main_py_thread is None or not self.run_main_py_thread.is_alive():
            self.run_main_py_thread = threading.Thread(target=self.recognize_and_check_blinks)
            self.run_main_py_thread.start()

    def delete_last_row(self):
        file_path = "check.csv"
        with open(file_path, "r") as file:
            rows = file.readlines()

        if rows:

            rows = rows[:-1]

            with open(file_path, "w", newline='') as file:
                writer = csv.writer(file)
                for row in rows:
                    file.write(row)







    def open_manual_window(self):
        self.manual_window = tk.Toplevel(self.window)
        self.manual_window.title("Manual Attendance")
        
        # chỗ hiện tên nhập
        self.name_entry = tk.Entry(self.manual_window, width=40, font=("Courier", 16), border=5)
        self.name_entry.grid(row=0, column=0, columnspan=10, padx=10, pady=10)
        
        # bàn phím
        self.open_virtual_keyboard(self.manual_window)
        
        # nút in out
        self.in_button = tk.Button(self.manual_window, text="IN", command=lambda: self.manual_check("IN"))
        self.out_button = tk.Button(self.manual_window, text="OUT", command=lambda: self.manual_check("OUT"))
        self.in_button.grid(row=5, column=0, columnspan=5, pady=5)
        self.out_button.grid(row=5, column=5, columnspan=5, pady=5)
    def open_virtual_keyboard(self, parent_window):
        # Entry box for the virtual keyboard
        e = tk.Entry(parent_window, width=40, font=("Courier", 16), border=5)
        e.grid(row=0, column=0, columnspan=10, padx=10, pady=10)

        def Button_click(letter):
            current = e.get()
            e.delete(0, tk.END)
            e.insert(0, current + letter)

        letters = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
        ]

        for row_index, row in enumerate(letters, start=1):
            for col_index, letter in enumerate(row):
                tk.Button(parent_window, text=letter, width=5, height=2, font=("Courier", 14),
                          command=lambda l=letter: Button_click(l)).grid(row=row_index, column=col_index, padx=2, pady=2)

        tk.Button(parent_window, text="Space", width=40, height=2, font=("Courier", 14),
                  command=lambda: Button_click(" ")).grid(row=4, column=0, columnspan=10, pady=5)

        tk.Button(parent_window, text="OK", command=lambda: self.name_entry.insert(tk.END, e.get()), width=10).grid(row=5, column=0, columnspan=10, pady=5)




    def manual_check(self, status):
        name = self.name_entry.get()
        self.log_attendance(name, status)
        self.manual_window.destroy()






    def recognize_and_check_blinks(self):


        video_capture = cv2.VideoCapture(self.video_source)         #mở webcam
        if not video_capture.isOpened():
            print("VIDEO SOURCE ERROR")
            return

        detector = dlib.get_frontal_face_detector()                         #phát hiện khuôn mặt
        predictor = dlib.shape_predictor(shape_predictor_path)              #dự đoán dựa trên facial_landmark
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]       #lấy giá trị các điểm mốc trên mắt
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        EYE_AR_THRESH = 0.3                                                 #ngưỡng EAR nhắm mắt
        EYE_AR_CONSEC_FRAMES = 3                                            #số frame mắt cần nhắm
        COUNTER = 0                                                         #biến đếm khung nhắm
        TOTAL = 0                                                           #biến tổng khung nhắm

        with open(knn_model_path, 'rb') as f:                               #tải mô hình knn và nhãn
            knn_clf, known_labels = pickle.load(f)
            print(f"Loaded known labels: {known_labels}")

        while True:                                                         #xử lý từng khung hình (k1)
            ret, frame = video_capture.read()
            if not ret:
                print("Error 2")
                break

            (frame_height, frame_width) = frame.shape[:2]                           #kích thước khung hình (giả sử 1600x900)
            center_x, center_y = frame_width // 2, frame_height // 2                #tâm khung hình (giả sử 800x450)
            crop_width, crop_height = 300, 300                                      #300x300
            start_x = center_x - (crop_width // 2)                                  
            start_y = center_y - (crop_width // 2)
            end_x = start_x + crop_width
            end_y = start_y + crop_height
            cropped_frame = frame[start_y:end_y, start_x:end_x]                     #khung hình bị crop (k2) (từ tâm khung hình lấy 300x300)

            frame_copy = cv2.resize(cropped_frame, (0, 0), fx=0.25, fy=0.25)        #copy khung hình và resize từ 300 thành 75 (k3)
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)                #BGR thành RGB để nhận diện gương mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                          #copy khung hình (k1) và đổi sang RGB (k4)
            rects = detector(gray, 0)                                               #nhận diện gương mặt trên (k4) để vẽ bbox
            predictions = self.predict(frame_copy, knn_clf=knn_clf)                 #đoán label từ gương mặt của (k3)

            for (pred_name, (top, right, bottom, left), rec) in predictions:        #từng dự đoán, ta có:
                print(f"Predicted: {pred_name}")                                    
                top *= 4                                                            #ảnh dự đoán là 25% ảnh gốc, x4 để thành 100%
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(cropped_frame, (left, top), (right, bottom), (0, 255, 255), 2)    #vẽ bbox trên (k2)
                cv2.putText(cropped_frame, f"{pred_name}", (left, top - 10),                    #hiển thị tên dự đoán
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                self.last_pred_name = pred_name                                                 #lưu dự đoán
                if pred_name != "unknown" and len(rects) > 0:                                   #dự đoán khác unknown và gương mặt > 0
                    shape = predictor(gray, rects[0])                                           #xác định 68 điểm trên mặt
                    shape = face_utils.shape_to_np(shape)                                       #chuyển numpy
                    leftEye = shape[lStart:lEnd]                                                #tách  điểm của mắt
                    rightEye = shape[rStart:rEnd]
                    leftEAR = self.eye_aspect_ratio(leftEye)                                    #tính EAR từng mắt
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    if ear < EYE_AR_THRESH:                                                     #chớp thì ear nhỏ lại, nhắm mắt xảy ra
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:                                     #nhắm được 3 frame, tính 1 lần chớp mắt
                            TOTAL += 1
                            self.prediction_label.config(text=f"Predicted: {pred_name} blinked!")
                            self.set_light('green')
                            print(f"{pred_name} recognized and blinked!")
                            self.log_blink(pred_name)
                            self.window.after(2000, self.reset_recognition)
                        COUNTER = 0

            cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)              #chuyển lại BGR để hiện lên GUI
            img = Image.fromarray(cropped_frame_rgb)                                        #chuyển numpy thành PIL
            imgtk = ImageTk.PhotoImage(image=img)                                           #PIL sang đối tượng trên GUI
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    #hàm dự đoán
    def predict(self, img, knn_clf=None, threshold=0.5, distance_threshold=0.5):
        face_box = face_recognition.face_locations(img)                                         #xác đinh vị trí gương mặt, trả về tọa độ(top right bot left)
        if len(face_box) == 0:                                                                  #nếu không có gương mặt thì thôi
            return []

        faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_box)   #mã hóa embedding gương mặt

        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)                  #với mỗi mã hóa gương mặt, tìm 1 điểm gần nhất, lưu vào list
        matches = [closest_distances[0][i][0] <= threshold for i in range(len(face_box))]       #khoảng các nhỏ nhất gương mặt thứ i với dataset, so với thresshold

        predictions = knn_clf.predict(faces_encodings)                  #dự đoán đựa trên mã hóa
        predictions = [str(pred) for pred in predictions]               #đảm bảo dạng chuỗi

        #Unknown nếu giá trị đoán nhỏ hơn thresshold
        predictions = ["unknown" if not match or closest_distances[0][i][0] > distance_threshold else pred #không có mặt khớp hoặc lớn hơn thresshold
                    for i, (pred, match) in enumerate(zip(predictions, matches))]

        return [(pred, loc, rec) for pred, loc, rec in zip(predictions, face_box, matches)]
        #trả về dự đoán, vị trí bbox, kết quả khớp hay không




    #đặt điều kiện sớm trễ auto va manual
    def log_attendance(self, name, status):
        now = datetime.now().strftime('%H:%M')
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            records = [row for row in reader if row[0] == name]

    
        if len(records) % 2 == 0:
            status = '+'
        else:
            status = '-'

        if len(records) == 0:
            early_or_late = "early" if "07:00" <= now <= "07:30" else "late"
            with open(file_path, "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, status, now, early_or_late])
        else:
            with open(file_path, "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, status, now])


    def log_blink(self, pred_name):
        now = datetime.now().strftime('%H:%M')
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            records = [row for row in reader if row[0] == pred_name]

        print(f"Records for {pred_name}: {records}")

        if len(records) % 2 == 0:
            status = '+'
        else:
            status = '-'

        if len(records) == 0:
            early_or_late = "early" if "07:00" <= now <= "07:30" else "late"
            with open(file_path, "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([pred_name, status, now, early_or_late])
        else:
            with open(file_path, "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([pred_name, status, now])




    def reset_recognition(self):
        self.prediction_label.config(text="Predicted: ")
        self.last_pred_name = None
        self.set_light('red')

    #Công thức EAR
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

root = tk.Tk()
root.configure(bg="#add8e6")
app = WebcamGUI(root, "GUI")
