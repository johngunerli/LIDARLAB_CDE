# CDE
Coordinate distance estimation using MediaPipe



https://github.com/johngunerli/LIDAR_PRJ/assets/33205097/138663e4-def9-4c06-8643-93ab80e73cbf





# MediaPipe Pose Estimation and Distance Measurement

This Python script uses OpenCV and MediaPipe to estimate the position and distance of a person from the camera based on shoulder landmarks. It calculates focal length based on the field of view and resolution width and records the horizontal offset and distance of the person from the camera.

## Features

- **Focal Length Calculation**: Dynamically calculates the focal length in pixels based on the provided field of view and resolution width.
- **Position and Distance Estimation**: Estimates the horizontal offset and distance from the camera using shoulder landmarks detected by MediaPipe Pose.
- **Real-Time Video Processing**: Processes video frames in real-time to detect pose landmarks and calculate distances.
- **Data Recording**: Records X and Y coordinates representing the horizontal offset and distance, respectively, over a specified number of instances.

## Requirements

- Python 3
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

## Usage

1. **Setup**: Install the required libraries using `pip install cv2 mediapipe numpy`.
2. **Run the Script**: Execute the script to start the pose estimation and distance measurement. The script will capture video from the default camera.
3. **View Output**: The script displays the video feed with pose landmarks. It prints and saves the recorded X and Y coordinates as a NumPy array.

## Customization (you need to do this based on hardware)

- **Field of View (FOV)**: Set the camera's field of view in degrees. Default is 60 degrees.
- **Resolution Width (RW)**: Set the resolution width of the camera in pixels. Default is 1280 pixels.
- **Known Width (KNOWN_WIDTH)**: Set the known width of the object (e.g., shoulder width) in centimeters. Default is 35 cm.

## Output

- Displays real-time video with pose landmarks and calculated distances.
- Prints the X and Y coordinates in the console.
- Saves the coordinates as a NumPy array in `xy_coordinates.npy`.

## Note
Ensure that the camera used matches the FOV and resolution width settings for accurate measurements.




https://www.section.io/engineering-education/multi-person-pose-estimator-with-python/