import os
import random
import shutil
# Load mô hình YOLOv5 đã được train
import torch

test_folder = 'custom_data\\test\\images'
target_folder = 'custom_data\\test\\predict\\'
model = torch.hub.load('ultralytics/yolov5','custom',path = 'yolov5\\face_detection.pt')
labels_test = 'custom_data\\test\\labels'
labels_predict = 'custom_data\\test\\predict\\'
def save_predict():
    images_test = os.listdir(test_folder)
    # Thực hiện dự đoán trên tập hình ảnh kiểm tra
    for image in images_test:
        # Thực hiện dự đoán trên tập hình ảnh kiểm tra
        img_test_path = os.path.join(test_folder,image)
        results = model(img_test_path)
        # Get the detected objects
        objects = results.pred[0]
        predict_str = ''
        image_path = os.path.join(target_folder, image.replace('.jpg','.txt'))
        # Lặp qua từng mẫu trong tập hình ảnh kiểm tra
        for ob in objects:
            # Tạo tên tệp tin .txt cho mẫu thứ i (ví dụ: result_0.txt, result_1.txt, ...)
            # Đường dẫn đến hình ảnh kiểm tra
            predict_str += ' '.join(map(str,ob.tolist()))+'\n'

            # Mở tệp tin để lưu kết quả
        with open(image_path, 'w') as file:
            file.write(predict_str + '\n')

        print(f"Saved {image_path}")

def setSampleArrays(input_file_path):
    a = []
    i = 0
    with open(input_file_path, 'r') as file:
        for line in file:
                # Tách các giá trị trong dòng bằng khoảng trắng và chuyển đổi chúng thành số thực
            a.append(i)
            i+=1
    return a

def averaged(a,n):
    s = 0
    for x in a:
        s+=x
    return s/n

def cal():
    acc,pre,rec,f1 = [],[],[],[]
    labels = os.listdir(labels_test)
    num_sample = len(labels)
    # Thực hiện dự đoán trên tập hình ảnh kiểm tra
    for label in labels:
        # Thực hiện dự đoán trên tập hình ảnh kiểm tra
        label_test_path = os.path.join(labels_test,label)
        label_predict_path = os.path.join(labels_predict,label)
        a = setSampleArrays(label_test_path)
        b = setSampleArrays(label_predict_path)
        c = set(a) & set(b)
        acc.append(1 if len(a) == len(b) and len(b) == len(c) else 0)
        pre.append(len(c)/len(b) if len(b) != 0 else 0)
        rec.append(len(c)/len(a) if len(a) != 0 else 0) 
        # Mở tệp tin và đọc từng dòng
    Ac = averaged(acc,num_sample)
    Pre = averaged((pre),num_sample)
    Rec = averaged(rec,num_sample)
    F1_score = (2*Pre*Rec)/(Pre + Rec)
    print('Ac: ',Ac)
    print('Pre: ',Pre)
    print('Rec: ',Rec)
    print('F1_score: ',F1_score)
    
        
        
