import cv2
import torch
import numpy as np
import os


def save_face(face, index):
    output_dir = 'yolov5\\face_images'
    #  # Tạo thư mục nếu nó không tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Resize the face to 640x640
    face_resized = cv2.resize(face, (640, 640))
    # Tạo đường dẫn lưu trữ ảnh khuôn mặt
    face_filename = os.path.join(output_dir, f'face_{index}.jpg')
    cv2.imwrite(face_filename, face_resized)
    print(f"Saved {face_filename}")
    
# # Load pre-trained YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can choose different model sizes like 'yolov5m' or 'yolov5l'
model = torch.hub.load('ultralytics/yolov5','custom',path = 'yolov5\\face_detection.pt')
eyes_state_model = torch.hub.load('ultralytics/yolov5','custom',path = 'yolov5\\eyes_state.pt')
names = model.module.names if hasattr(model,'module') else model.names
# Initialize webcam
def predict_cam():
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam, you can change it to another camera index if you have multiple cameras

    i = 0
    drowsiness = 0
    alarm_max_frame = 5
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Perform inference on the frameq
        results = model(frame)

        # Get the detected objects
        objects = results.pred[0]

        # Draw bounding boxes on the frame
        isOK = False
        for obj in objects:
            x1, y1, x2, y2, conf, label = obj.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Label: {names[int(label)]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Crop the face from the frame
            face = frame[y1:y2, x1:x2]
            
            save_face(face,i)
            i+=1
            ##=======================eyes state========================
        #     eyes = eyes_state_model(face)
        #     state = eyes.pred[0]
        #     for st in state:
        #         x1e, y1e, x2e, y2e, confe, labele = st.tolist()
        #         x1e, y1e, x2e, y2e = map(int, [x1e, y1e, x2e, y2e])
        #         cv2.rectangle(face, (x1e, y1e), (x2e, y2e), (0, 255, 0), 2)
        #         cv2.putText(face, f'state: {names[int(labele)]}', (x1e, y1e - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #         # save_face(face, i)
        #         print(f'state: {names[int(labele)]}')
        #         if(labele == 1):
        #             drowsiness += 1
        #         else: drowsiness = 0
        #         i+=1
        #         isOK = True
        # # Display the frame
        # if isOK == False:
        #     print('no detect!')
        # if drowsiness > alarm_max_frame:
        #     print('Wake up now, you are drowsiness')
        cv2.imshow('Face Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
    
predict_cam()

def predict_img():
    path_img = 'custom_data\\test\images\\15-08608-001_jpg.rf.a908e7d5b9da2ffe5560973f95a3527c.jpg'
    frame = cv2.imread(path_img)
    results = model(frame)

        # Get the detected objects
    objects = results.pred[0]
    index = 1
        # Draw bounding boxes on the frame
    for obj in objects:
        x1, y1, x2, y2, conf, label = obj.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        nameOb = names[int(label)] + ' '+ str(index)
        cv2.putText(frame, f'Label: {nameOb}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Crop the face from the frame
        face = frame[y1:y2, x1:x2]
        index+=1
        print(x1,y1,x2,y2,label)
        # save_face(face, i)
        # i+=1
        # Display the frame
    cv2.imshow('Face Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()