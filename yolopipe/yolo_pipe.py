import cv2
import numpy as np
import math
import os
import mediapipe as mp

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

def calculate_focal_length_in_pixels(resolution_width, FOV_degrees):
    FOV_radians = FOV_degrees * (math.pi / 180)
    focal_length_pixels = resolution_width / (2 * math.tan(FOV_radians / 2))
    return focal_length_pixels

# Load YOLO
net = cv2.dnn.readNet("../yolo_config/yolov3.weights", "../yolo_config/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load image
image = cv2.imread("image.jpg")
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

# Create a directory to save processed images
os.makedirs("yolo_person_processed", exist_ok=True)
os.makedirs("yolo_person_detections", exist_ok=True)

# Load processed person images
processed_images = []
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        crop_img = image[y:y+h, x:x+w]
        cv2.imwrite(f'yolo_person_detections/person_{i}.jpg', crop_img)

# now load these images, and do the mediapipe stuff

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

FOV = 60  # Field of View in degrees
RW = 1280  # Resolution Width in pixels
KNOWN_WIDTH = 35  # Approximate shoulder width of an adult in cm
FOCAL_LENGTH = calculate_focal_length_in_pixels(RW, FOV)

xy_coordinates = []

image_directory = "yolo_person_detections"  # Update this with the path to your cropped images
processed_directory = "yolo_person_processed"  # Update this with the path to save processed images

# Create the processed images directory if it doesn't exist
os.makedirs(processed_directory, exist_ok=True)
os.makedirs(image_directory, exist_ok=True)

for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_directory, filename)
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

            processed_image_path = os.path.join(processed_directory, filename)
            cv2.imwrite(processed_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Load the original image again
image = cv2.imread("image.jpg")

# Paste processed person images back onto the original image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        processed_person_image_path = os.path.join(processed_directory, f"person_{i}.jpg")
        processed_person_image = cv2.imread(processed_person_image_path)
        processed_person_image = cv2.resize(processed_person_image, (w, h))
        image[y:y+h, x:x+w] = processed_person_image

# Save the final image
cv2.imwrite("final_image.jpg", image)

np_xy_coordinates = np.array(xy_coordinates)
print("X and Y coordinates:", np_xy_coordinates)
