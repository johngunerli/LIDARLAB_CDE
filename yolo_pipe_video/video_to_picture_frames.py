# write a program to get a video and convert it to frames, saved in a folder

import cv2
import os
# def video_to_frames(video_path, frames_folder):
#     # Opens the Video file
#     os.makedirs(frames_folder, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     i = 0
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == False:
#             break
#         cv2.imwrite(f'{frames_folder}/frame_{i}.jpg', frame)
#         i += 1
 
#     cap.release()
#     cv2.destroyAllWindows()
    
# video_to_frames("vids/testVid.mov", "frames")  # Uncomment this line to run the function

os.makedirs("detected_video", exist_ok=True)

import os
from video_yolo_pipe import process_image

frames_folder = 'frames'

# Loop through each file in the frames folder
for file in os.listdir(frames_folder):
    file_path = os.path.join(frames_folder, file)
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        processed_image = process_image(file_path)
        # Now you can do something with the processed image
        cv2.imwrite(f'detected_video/{file}', processed_image)
    