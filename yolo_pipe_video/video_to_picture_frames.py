import cv2
import os
from video_yolo_pipe import process_image

# # Define video path and output directory
# video_path = "./vids/testVideo.mp4"
# output_dir = "frames"

# # Create output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Open video capture object
# cap = cv2.VideoCapture(video_path)

# # Check if video opened successfully
# if not cap.isOpened():
#     print("Error opening video")
#     exit(1)

# # Get frame rate
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Define frame count
# frame_count = 0

# # Read frame by frame
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Check if frame read successfully
#     if not ret:
#         print("Video finished")
#         break

#     # Save frame as image in output directory
#     frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
#     cv2.imwrite(frame_filename, frame)
#     # Increase frame count
#     frame_count += 1

# # Release capture object
# cap.release()

# print(f"Total frames saved: {frame_count}")
# print(f"Frame rate: {fps:.2f} FPS")

# Process saved frames
os.makedirs("detected_video", exist_ok=True)

frames_folder = 'frames'

# Loop through each file in the frames folder
for frame_number, file in enumerate(os.listdir(frames_folder)):
    file_path = os.path.join(frames_folder, file)
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        processed_image = process_image(file_path, frame_number)
        # Now you can do something with the processed image
        cv2.imwrite(f'detected_video/{file}', processed_image)

# Combine processed frames back into a video
input_dir = "detected_video"
output_video = "combined_video.mp4"

# Get all images in the input directory
images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jpg")]

# Check if there are any images
if len(images) == 0:
    print("No images found in the input directory")
    exit(1)

# Read the first image to get the frame size
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define video writer object
video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))

# Write each image to the video
for image in images:
    frame = cv2.imread(image)
    video_writer.write(frame)

# Release video writer object
video_writer.release()

print(f"Combined video saved as: {output_video}")
