import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to estimate distance using shoulder landmarks.
def estimate_distance(image_width, landmark1, landmark2):
    KNOWN_WIDTH = 35  # Approximate shoulder width of an adult in cm
    FOCAL_LENGTH = 800  # Focal length of the camera in pixels, needs calibration

    # Calculate the pixel distance between landmarks
    pixel_distance = ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5 * image_width

    # Calculate distance
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_distance
    return distance

# List to store distance measurements
distance_measurements = []

# Timing and counting variables
start_time = time.time()
time_interval = 0.4  # Time interval in seconds
measurement_count = 0  # Counter for the number of measurements
max_measurements = 8  # Maximum number of measurements

# Capture video
cap = cv2.VideoCapture(0)

while cap.isOpened() and measurement_count < max_measurements:
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmark1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        landmark2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        ih, iw, ic = image.shape
        distance = estimate_distance(iw, landmark1, landmark2)

        if time.time() - start_time >= time_interval:
            distance_measurements.append(distance)
            measurement_count += 1
            start_time = time.time()

        # Draw landmarks, connections, and distance text
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(image, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Pose', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()

# Save the recorded distances to a text file
with open("distance_measurements.txt", "w") as file:
    for distance in distance_measurements:
        file.write(f"{distance}\n")

print("Distance measurements saved to distance_measurements.txt")
