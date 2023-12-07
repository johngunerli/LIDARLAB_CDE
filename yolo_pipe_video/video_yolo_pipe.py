import cv2
import numpy as np
import math
import os
import mediapipe as mp


# Mediapipe processing
FOV = 70  # Field of View in degrees
RW = 1280  # Resolution Width in pixels
KNOWN_WIDTH = 35  # Approximate shoulder width of an adult in cm


def calculate_focal_length_in_pixels(resolution_width, FOV_degrees):
    FOV_radians = FOV_degrees * (math.pi / 180)
    focal_length_pixels = resolution_width / (2 * math.tan(FOV_radians / 2))
    return focal_length_pixels

FOCAL_LENGTH = calculate_focal_length_in_pixels(RW, FOV)

def estimate_position_and_distance(image_width, image_height, landmark1, landmark2, focal_length, known_width):
    pixel_distance = math.hypot((landmark1.x - landmark2.x) * image_width, (landmark1.y - landmark2.y) * image_height)
    distance = (known_width * focal_length) / pixel_distance
    avg_x = (landmark1.x + landmark2.x) / 2 * image_width
    avg_y = (landmark1.y + landmark2.y) / 2 * image_height
    center_x = image_width / 2
    pixel_offset = avg_x - center_x
    distance_per_pixel = math.tan(math.radians(FOV / 2)) * 2 * distance / image_width
    cm_offset = pixel_offset * distance_per_pixel
    return cm_offset, distance, (int(avg_x), int(avg_y))



def process_image(input_image_name):
    # Load YOLO
    net = cv2.dnn.readNet("../yolo_config/yolov3.weights", "../yolo_config/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Load image
    image = cv2.imread(input_image_name)
    height, width, channels = image.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
            if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 corresponds to persons
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

    # Create directories
    processed_directory = "yolo_person_processed"
    detections_directory = "yolo_person_detections"
    os.makedirs(processed_directory, exist_ok=True)
    os.makedirs(detections_directory, exist_ok=True)

    # YOLO detection and cropping code...
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            crop_img = image[y:y+h, x:x+w]
            cv2.imwrite(f'{detections_directory}/person_{i}.jpg', crop_img)

    
    xy_coordinates = []

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    for filename in os.listdir(detections_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(detections_directory, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image)

            if results.pose_landmarks:
                landmark1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                landmark2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                
                ih, iw, ic = image.shape
                cm_offset, distance, center_point = estimate_position_and_distance(iw, ih, landmark1, landmark2, FOCAL_LENGTH, KNOWN_WIDTH)
                xy_coordinates.append((cm_offset, distance))

                mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.circle(image, center_point, 5, (255, 0, 0), -1)
                cv2.putText(image, f"Offset: {cm_offset:.2f}cm, Distance: {distance:.2f}cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                print(f"Offset: {cm_offset:.2f}cm, Distance: {distance:.2f}cm")
                # save these the offset and distance under filename.npy 
                os.makedirs("xy_data", exist_ok=True)
                np.save(f"xy_data/{filename}.npy", np.array([cm_offset, distance]))
                

                processed_image_path = os.path.join(processed_directory, filename)
                cv2.imwrite(processed_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # Load the original image again
                image = cv2.imread(input_image_name)
                # Paste processed person images back onto the original image
                for i in range(len(boxes)):
                    if i in indexes:                        
                        # if the file does not exist, skip it.
                        try: 
                            x, y, w, h = boxes[i]
                            processed_person_image_path = os.path.join(processed_directory, f"person_{i}.jpg")
                            processed_person_image = cv2.imread(processed_person_image_path)
                            processed_person_image = cv2.resize(processed_person_image, (w, h))
                            image[y:y+h, x:x+w] = processed_person_image 
                        except cv2.error:
                            pass
                return image
                