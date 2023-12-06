import cv2
import numpy as np
import math

# Function to calculate focal length in pixels from the field of view and resolution width


def calculate_focal_length_in_pixels(resolution_width, FOV_degrees):
    FOV_radians = FOV_degrees * (math.pi / 180)
    focal_length_pixels = resolution_width / (2 * math.tan(FOV_radians / 2))
    return focal_length_pixels


# Known parameters (to be calibrated for your specific setup)
FOV = 50  # Field of View in degrees
KNOWN_WIDTH_CM = 35  # Approximate shoulder width of an adult in cm

# Load YOLO
net = cv2.dnn.readNet("../yolo_config/yolov3.weights", "../yolo_config/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1]
                 for i in net.getUnconnectedOutLayers().flatten()]

# Load image
image = cv2.imread("image.jpg")
height, width, channels = image.shape

# Calculate the focal length in pixels based on the camera's field of view
focal_length_px = calculate_focal_length_in_pixels(width, FOV)

# Detecting objects
blob = cv2.dnn.blobFromImage(
    image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
        if confidence > 0.5:
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
for i in range(len(boxes)):
    if i in indexes:
        box = boxes[i]
        x, y, w, h = box
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        # Estimate the distance from the camera to the person
        distance_cm = (KNOWN_WIDTH_CM * focal_length_px) / w

        image_width_real_cm = 2 * \
            (distance_cm * math.tan(math.radians(FOV / 2)))
        center_x_cm = (center_x - (width / 2)) * (image_width_real_cm / width)

        # Convert the center_y to cm assuming the height of the image represents 2*distance_cm*tan(FOV/2)
        center_y_cm = ((center_y - height / 2) * (2 * distance_cm *
                       math.tan(math.radians(FOV / 2))) / height) * 10
        # i believe that there is a problem with the distance measurement of y, not with x though.

        # Save the offset and distance in cms for each detected person
        position_data_cm = np.array([center_x_cm, center_y_cm, distance_cm])
        np.save(f'person_{i}_position_cm.npy', position_data_cm)

        # Draw bounding box and label on the image with the (x, y) coordinates in cms
        coordinates_text = f"X: {center_x_cm:.2f}cm, Y: {center_y_cm:.2f}cm"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, coordinates_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image with annotations
cv2.imwrite("image_with_annotations.jpg", image)
