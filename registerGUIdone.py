import tkinter as tk                                    #Thư viện tạo GUI
from tkinter import ttk
import cv2                                              #Thư viện xử lý ảnh
import threading                                        #Chạy song song việc
import os                                               #Quản lý đường dẫn, địa chỉ
from PIL import Image, ImageTk                          #Quản lý hình ảnh trên GUI
import numpy as np                                      #Tạo mảng
from sklearn import neighbors                           #Thư viện tạo model KNN
import face_recognition                                 #Thư viện nhận diện gương mặt
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle                                           #Lưu dữ liệu cho model
import imgaug.augmenters as iaa                         #Tăng cường dữ liệu
#from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor       #Quản lý công việc đồng thời 

seq = iaa.Sequential([iaa.Multiply((0.8, 1.2))])        #Đổi độ sáng ảnh 80% đến 120%

class App:                                              #Tạo ứng dụng
    def __init__(self, root):                           #Tạo giao diện
        self.root = root                                
        self.root.title("Dang ki")
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", self.exit_fullscreen)

        #Tạo các giá trị nhớ tạm
        self.video_source = 0
        self.vid = None
        self.recording = False
        self.video_filename = ""
        self.name = ""

        #Load ảnh cho nút và chỉnh cỡ
        original_image = Image.open("circle.png")
        self.loadimage = ImageTk.PhotoImage(original_image.resize((75, 75)))

        #Tạo khung cho webcam
        self.video_label = tk.Label(self.root, bd=2, relief="solid", highlightbackground="black", highlightthickness=2)
        self.video_label.place(relx=0.5, rely=0.5, anchor='center', width=400, height=400)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=300, mode='determinate')
        self.progress.place(relx=0.5, rely=0.5, anchor='n', y=240)

        #Tạo các nút và chỉ định công dụng
        self.name_button = tk.Button(self.root, text="NAME", bg="pink", command=self.open_virtual_keyboard, image=self.loadimage, compound='center')
        self.register_button = tk.Button(self.root, text="REGISTER", bg="blue", command=self.start_webcam, image=self.loadimage, compound='center')
        self.record_button = tk.Button(self.root, text="RECORD", bg="green", command=self.start_recording, image=self.loadimage, compound='center')
        self.convert_button = tk.Button(self.root, text="CONVERT", bg="orange", command=self.convert_video_to_images, image=self.loadimage, compound='center')
        self.train_button = tk.Button(self.root, text="TRAIN", bg="purple", command=self.train_model, image=self.loadimage, compound='center')
        self.close_button = tk.Button(self.root, text="CLOSE", bg="red", command=self.close, image=self.loadimage, compound='center')

        #Vị trí các nút
        self.name_button.place(x=-250, y=-160, relx=0.5, rely=0.5, anchor='center')
        self.register_button.place(x=-250, y=0, relx=0.5, rely=0.5, anchor='center')
        self.record_button.place(x=-250, y=160, relx=0.5, rely=0.5, anchor='center')
        self.convert_button.place(x=250, y=-160, relx=0.5, rely=0.5, anchor='center')
        self.train_button.place(x=250, y=0, relx=0.5, rely=0.5, anchor='center')
        self.close_button.place(x=250, y=160, relx=0.5, rely=0.5, anchor='center')

    #Khởi động webcam
    def start_webcam(self):
        if self.vid is None:                                                                    #Nếu webcam chưa mở
            self.vid = cv2.VideoCapture(self.video_source)                                      #Mở webcam
            self.video_label.place(relx=0.5, rely=0.5, anchor='center', width=400, height=400)  #chọn ví trị webcam trên GUI
            self.update_frame()

    #Cập nhật khung hình
    def update_frame(self):
        if self.vid.isOpened():                                                 #nếu webcam đã mở
            ret, frame = self.vid.read()                                        #đọc khung hình từ webcam
            if ret:                                                             #nếu đọc được khung hình
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                  #chuyển đổi BGR sang RGB để xử lý ảnh
                height, width, _ = frame.shape                                  #lấy cao, rộng và kênh màu
                aspect_ratio = width / height                                   #tỉ lệ khung hình (đảm bảo xử lý ảnh xong không biến dạng ảnh)
                new_width = 400                                                 #đặt chiều ngang ảnh
                new_height = int(new_width / aspect_ratio)                      #tính chiều dọc ảnh

                if new_height > 400:
                    new_height = 400
                    new_width = int(new_height * aspect_ratio)

                frame = cv2.resize(frame, (new_width, new_height))              #thay đổi cao rộng frame về giá trị chọn
                canvas = Image.new("RGB", (400, 400), (255, 255, 255))          #tạo khung cho frame để hiện lên GUI
                canvas.paste(Image.fromarray(frame), ((400 - new_width) // 2, (400 - new_height) // 2)) #chuyển frame từ giá trị numpy sang ảnh và đặt vào khung

                self.photo = ImageTk.PhotoImage(image=canvas)                   #chuyển khung(1) frame thành ảnh hiện lên trên GUI
                self.video_label.configure(image=self.photo)                    #đặt vào khung đã tạo sẵn
                self.video_label.image = self.photo                             #đè khung(1) lên khung

            self.video_label.after(10, self.update_frame)                       #mỗi 10 milisec cập nhật khung 1 lần

    #Tắt chương trình
    def close(self):
        if self.vid is not None:
            self.vid.release()
        self.root.destroy()









    #chuẩn bị ghi hình
    def start_recording(self):
        if not self.recording and self.vid is not None:                         #nếu webcam mở và chưa bắt đầu ghi hình
            self.recording = True                                               #ghi hình
            self.record_thread = threading.Thread(target=self.record_video)     #tạo luồng để ghi hình(chạy song song GUI)   
            self.record_thread.start()                                          #bắt đầu luồng ghi hình

    #ghi hình
    def record_video(self):
        dataset_dir = 'dataset'                                         #địa chỉ các thư mục dataset
        if not os.path.exists(dataset_dir):                             #nếu chưa có thư mục thì tạo
            os.makedirs(dataset_dir)

        if self.name:                                                   #nếu đã đặt tên cho người đăng kí, lấy tên đó để lưu video
            self.video_filename = f"{self.name}.mp4"
        else:                                                           #nêu không thì tự tạo tên không trùng với các video đã lưu
            existing_videos = [f for f in os.listdir(dataset_dir) if f.endswith('.mp4')]
            new_video_id = str(len(existing_videos) + 1).zfill(2)
            self.video_filename = f"{new_video_id}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                        #định dạng mp3
        out = cv2.VideoWriter(os.path.join(dataset_dir, self.video_filename), fourcc, 30.0, (640, 480)) #output là video lưu ở thư mục dataset, 30fps, 640x480



        #số lượng dataset mong muốn
        frame_limit = 40




        #đang ghi hình thì hiện "Recording"
        status_label = tk.Label(self.root, text="Recording...", bg="white")
        status_label.place(relx=0.5, rely=0.4, anchor='center')



        #Đếm số lượng khung hình đã ghi
        frame_count = 0
        while frame_count < frame_limit and self.vid.isOpened():    #nếu ít hơn limit
            ret, frame = self.vid.read()
            if ret:                                                 #đọc thành công hình thì lưu
                out.write(frame)
                frame_count += 1
            else:
                break

        self.recording = False                                       #tắt ghi hình
        out.release()
        #hiển thị Finish
        status_label.config(text="Finished")
        self.root.after(2000, status_label.destroy)




    #Chuyển video sang ảnh
    def convert_video_to_images(self):
        if self.vid is not None:                                            #tắt camera
            self.vid.release()
        if self.video_filename:                                             #kiểm tra sự tồn tại của video
            video_path = os.path.join('dataset', self.video_filename)       #vị trí video
            video_id = os.path.splitext(self.video_filename)[0]             #lấy tên video để đặt tên cho ảnh (tên_video, .mp4)

            image_folder = os.path.join('dataset', video_id)                #vị trí thư mục ảnh output (tạo nếu chưa có)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            cap = cv2.VideoCapture(video_path)                              #mở video để đọc
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_filename = f"{video_id}_{str(frame_count).zfill(3)}.jpg"  #đặt tên cho từng ảnh
                cv2.imwrite(os.path.join(image_folder, image_filename), frame)  #lưu ảnh đang xử lý
                frame_count += 1

            cap.release()                                                       #tắt video
            print(f"Video converted to {frame_count} images in folder '{image_folder}'")

    #Khởi động huấn luyện chương trình
    def train_model(self):
        self.training_thread = threading.Thread(target=self._train_model)
        self.training_thread.start()

    #Huấn luyện chương trình sử dụng model KNN
    def _train_model(self):

        #địa chỉ đường dẫn
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_save_path = os.path.join(base_path, "classifier", "trained_knn_model.clf")
        #dataset_dir = os.path.join(base_path,'..', "dataset")
        dataset_dir = os.path.join(base_path, "dataset")

        #nếu đã có model
        if os.path.exists(model_save_path):
            with open(model_save_path, 'rb') as f:              #mở model
                knn_clf, known_labels = pickle.load(f)
                existing_X = knn_clf._fit_X.tolist()            #lấy đặc điểm
                existing_y = known_labels.tolist()              #lấy label
        else:
            existing_X = []                                     #mảng mới
            existing_y = []
            
        #train data mới
        new_X, new_y, trained_folders = train_latest_folder(dataset_dir, verbose=True, update_progress=self.update_progress)    #từ dataset mới, lấy x,y và model mới
        
        #kết hợp dữ liệu mới và cũ
        X = existing_X + new_X
        y = existing_y + new_y

        #kiểm tra ảnh có đặc trưng và label không
        if len(X) == 0 or len(y) == 0:
            print("No features or labels collected. Cannot fit the model.")
            return

        #label thành chuỗi
        y = [str(label) for label in y]             #đảm bảo mảng y chứa chuỗi label
        print(f"Class labels: {set(y)}")

        #chuyển 2 mảng thành numpy để huấn luyện
        X = np.array(X)
        y = np.array(y)

        #khởi động train
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', weights='distance')
        #n_neighbors: mặc định là 5 nhưng giá trị sử dụng là đặc điểm gương mặt, chọn giá trị nhỏ hơn
        #algorithm: tập dữ liệu chiều vừa đến lớn, nhận diện gương mặt nên dùng ball_tree để tìm neighbor tốt hơn
        #weights: neighbors gần thì weight lớn
        #metric: mặc định là minkowski (Euclidean)

        #train classifier
        knn_clf.fit(X, y)

        #lưu model
        with open(model_save_path, 'wb') as f:
            pickle.dump((knn_clf, np.array(y)), f)
        #lưu model đã lưu folder nào
        save_trained_folders(trained_folders)
        print("Training complete, model saved.")

    #progress đã train
    def update_progress(self, processed_images, total_images):
        percentage = (processed_images / total_images) * 100        #công thức
        self.progress['value'] = percentage                         #cập nhật giá trị của pb
        self.root.update_idletasks()                                #cập nhật pb trên GUI đảm bảo không crash
        print(f"Training progress: {processed_images}/{total_images} ({percentage:.2f}%)")  #in để debug

    #bàn phím ảo
    def open_virtual_keyboard(self):
        self.keyboard_window = tk.Toplevel(self.root)                               #chạy bàn phím ảo đè GUI
        self.keyboard_window.title("Keyboard")                                      #tên bàn phím ảo

        e = tk.Entry(self.keyboard_window, width=40, font=("Courier", 16), border=5)#widget hiện chữ đã nhập
        e.grid(row=0, column=0, columnspan=10, padx=10, pady=10)                    #vị trí

        def Button_click(letter):                                                   #hiện chữ đã nhập lên widget
            current = e.get()
            e.delete(0, tk.END)
            e.insert(0, current + letter)

        letters = [                                                                 #chữ trên bàn phím
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
        ]

        for row_index, row in enumerate(letters, start=1):                                                                  #mỗi dòng lặp và từng dòng và vị trí dòng
            for col_index, letter in enumerate(row):                                                                        #mỗi giá trị số đặt cho từng chữ cái trên từng dòng
                tk.Button(self.keyboard_window, text=letter, width=5, height=2, font=("Courier", 14),                       #tạo nút
                        command=lambda l=letter: Button_click(l)).grid(row=row_index, column=col_index, padx=2, pady=2)

        tk.Button(self.keyboard_window, text="Space", width=40, height=2, font=("Courier", 14),         #dấu cách
                command=lambda: Button_click(" ")).grid(row=4, column=0, columnspan=10, pady=5)

        tk.Button(self.keyboard_window, text="OK", command=lambda: self.save_name(e.get()), width=10).grid(row=5, column=0, columnspan=10, pady=5)      #lưu tên khi ấn OK

    #lưu tên và đóng bàn phím ảo
    def save_name(self, name):
        self.name = name
        self.keyboard_window.destroy()

    #tắt fullscreen khi ấn esc
    def exit_fullscreen(self, event):
        self.root.attributes('-fullscreen', False)


def train_latest_folder(train_dir, verbose=False, update_progress=None):

    #khởi tạo và chuẩn bị đường dẫn
    X = []                                              #tạo danh sách chứa đặc trưng
    y = []                                              #tạo danh sách chứa label
    trained_folders = load_trained_folders()            #train tiếp trên file model đã train

    folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]   #vị trí các thư mục dataset
    untrained_folders = [f for f in folders if f not in trained_folders]                        #tìm thư mục chưa train

    if not untrained_folders:                                       #nếu đã train, bỏ qua
        print("No new folders found for training.") 
        return X, y, trained_folders

    latest_folder = max(untrained_folders, key=lambda folder: os.path.getmtime(os.path.join(train_dir, folder)))    #tìm thư mục mới nhất
    latest_folder_path = os.path.join(train_dir, latest_folder)

    if verbose:                                                         #kiểm tra đang train folder nào
        print("Training on the latest folder:", latest_folder)

    #hàm xử lý ảnh
    def process_image(img_path):
        image = face_recognition.load_image_file(img_path)              #load ảnh
        face_bounding_boxes = face_recognition.face_locations(image)    #xác định vị trí gương mặt để vẽ bbox
        print("Processing:", img_path)

        #số lượng bbox là 1
        if len(face_bounding_boxes) == 1:
            face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]                         #lấy giá trị embedding từ ảnh (face encoding) trả về giá trị từng điểm trên mặt
            augmented_image = seq(image=image)                                                                                          #tăng cường ảnh
            augmented_face_encoding = face_recognition.face_encodings(augmented_image, known_face_locations=face_bounding_boxes)[0]     #lấy giá trị embedding
            X.append(face_encoding)                 #bỏ các giá trị embedding vào X
            X.append(augmented_face_encoding)
            y.append(latest_folder)                 #bỏ label vào y, 2 lần vì có 2 lần X
            y.append(latest_folder)
            process_image.counter += 1              #đếm số ảnh đã xử lý
            update_progress(process_image.counter, total_images)    #cập nhật pb
            return face_encoding, augmented_face_encoding

    process_image.counter = 0                                   #reset đếm
    image_paths = image_files_in_folder(latest_folder_path)
    total_images = len(image_paths)                             #tìm tổng số ảnh cần train

    with ThreadPoolExecutor() as executor:                      #chạy nhiều thread
        list(executor.map(process_image, image_paths))

    update_progress(total_images, total_images)
    trained_folders.append(latest_folder)
    save_trained_folders(trained_folders)

    return X, y, trained_folders


#lưu tên folder đã train
def save_trained_folders(trained_folders, file_path="trained_folders.txt"):
    with open(file_path, 'w') as f:
        for folder in trained_folders:
            f.write(f"{folder}\n")

#load tên folder đã train
def load_trained_folders(file_path="trained_folders.txt"):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()