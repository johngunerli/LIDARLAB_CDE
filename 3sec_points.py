import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Function to calculate focal length in pixels from the field of view and resolution width
def calculate_focal_length_in_pixels(resolution_width, FOV_degrees):
    FOV_radians = FOV_degrees * (math.pi / 180)
    focal_length_pixels = resolution_width / (2 * math.tan(FOV_radians / 2))
    return focal_length_pixels

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to estimate horizontal offset and distance from the camera using shoulder landmarks.
def estimate_position_and_distance(image_width, image_height, landmark1, landmark2, focal_length, known_width):
    # Calculate the pixel distance between landmarks
    pixel_distance = math.hypot((landmark1.x - landmark2.x) * image_width, (landmark1.y - landmark2.y) * image_height)

    # Calculate distance from the camera
    distance = (known_width * focal_length) / pixel_distance

    # Calculate average position (central point between the shoulders in image coordinates)
    avg_x = (landmark1.x + landmark2.x) / 2 * image_width
    avg_y = (landmark1.y + landmark2.y) / 2 * image_height

    # Calculate horizontal offset from the center of the image in pixels
    center_x = image_width / 2
    pixel_offset = avg_x - center_x

    # Assuming a constant real-world field of view, calculate offset in cm (approximation)
    distance_per_pixel = math.tan(math.radians(FOV / 2)) * 2 * distance / image_width
    cm_offset = pixel_offset * distance_per_pixel

    return cm_offset, distance, (int(avg_x), int(avg_y))

# Constants provided by the user
FOV = 60  # Field of View in degrees
RW = 1280  # Resolution Width in pixels

# Constants
KNOWN_WIDTH = 35  # Approximate shoulder width of an adult in cm
FOCAL_LENGTH = calculate_focal_length_in_pixels(RW, FOV)  # Focal length in pixels, calculated dynamically

# List to store X and Y coordinates
xy_coordinates = []

# Counter for instances recorded
instances_recorded = 0
max_instances = 8  # Maximum number of instances to record

# Timing variables
start_time = time.time()
time_interval = 0.4  # Time interval in seconds

# Capture video
cap = cv2.VideoCapture(0)

while cap.isOpened() and instances_recorded < max_instances:
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmark1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        landmark2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        ih, iw, ic = image.shape
        cm_offset, distance, center_point = estimate_position_and_distance(iw, ih, landmark1, landmark2, FOCAL_LENGTH, KNOWN_WIDTH)

        if time.time() - start_time >= time_interval:
            x_coord = (cm_offset)
            y_coord = (distance)
            xy_coordinates.append((x_coord, y_coord))
            instances_recorded += 1
            start_time = time.time()

        # Draw landmarks, connections, and the center point
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.circle(image, center_point, 5, (255, 0, 0), -1)
        cv2.putText(image, f"Offset: {cm_offset:.2f}cm, Distance: {distance:.2f}cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Pose', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()

# Convert the list of coordinates to a NumPy array
np_xy_coordinates = np.array(xy_coordinates)
print("X and Y coordinates:", np_xy_coordinates)

# Optionally, save the NumPy array to a file
np.save('xy_coordinates.npy', np_xy_coordinates)
print("NumPy array saved to xy_coordinates.npy")
