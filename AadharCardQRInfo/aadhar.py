from PIL import Image
import cv2
import numpy as np
from model_images import cv2_to_pil

CONFIDENCE_THRESHOLD = 0.8
FRAME_ROTATION_ANGLE = 0

net = cv2.dnn.readNet("qrcode-yolov3-tiny_last.weights", "qrcode-yolov3-tiny.cfg")
classes = []
with open("qrcode.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def save_face(img_path):
    image = cv2.imread(img_path)
    face_img = image.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor = 1.2, minNeighbors = 5)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    face_img = face_img[y:y+h, x:x+w]
    filename = 'D:\T\SpitHackathon\src\images/a.jpeg'
    cv2.imwrite(filename, face_img)

    img = image[950:-950, :]
    height, width, channels = img.shape 
    h32 = 32*(height//32)
    w32 = 32*(width//32)


    blob = cv2.dnn.blobFromImage(img, 1/255, (h32, w32), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    qr_code = None
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            qr_code = img[y:y+h, x:x+w]
    
    qr_filename = 'D:\T\SpitHackathon\src\images/qr_code.jpeg'
    cv2.imwrite(qr_filename, qr_code)