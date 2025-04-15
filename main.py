import cv2
import sqlite3
from ultralytics import YOLO
from matplotlib import pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import datetime

# Initialize YOLO model with pre-trained weights
vehicle_model = YOLO('yolov5s.pt')

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('vehicles.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS vehicle_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
''')
conn.commit()

# Function to save license plate data to the database
def save_to_database(plate_number):
    timestamp = datetime.datetime.now()
    cursor.execute("INSERT INTO vehicle_data (plate, timestamp) VALUES (?, ?)", (plate_number, timestamp))
    conn.commit()

# Function to recognize license plates using Tesseract OCR
def recognize_plate(vehicle_frame):
    # Convert the frame to grayscale for better OCR results
    gray_frame = cv2.cvtColor(vehicle_frame, cv2.COLOR_BGR2GRAY)
    plate_number = pytesseract.image_to_string(gray_frame, config='--psm 8')
    return plate_number.strip()

# Function to save license plate data to a text file
def save_to_text_file(plate_number):
    with open('detected_plates.txt', 'a') as file:  # Open the file in append mode
        timestamp = datetime.datetime.now()
        file.write(f"{plate_number}, {timestamp}\n")  # Write the plate number and timestamp

# Update the detect_and_recognize function to save to the text file
def detect_and_recognize(frame):
    results = vehicle_model(frame)  # Run YOLO model on the frame
    for result in results:  # Loop through the list of Results objects
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):  # Extract bounding boxes and class IDs
            x1, y1, x2, y2 = map(int, box[:4])  # Convert bounding box coordinates to integers
            class_id = int(cls)  # Get the class ID
            if class_id == 2:  # Check if the detected object is a car (class ID 2 for YOLOv5)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle

                # Crop the vehicle frame for license plate detection
                vehicle_frame = frame[y1:y2, x1:x2]
                plate_number = recognize_plate(vehicle_frame)
                if plate_number:
                    save_to_database(plate_number)  # Save to database
                    save_to_text_file(plate_number)  # Save to text file
                    cv2.putText(frame, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    # Function to display the frame using OpenCV or Matplotlib (fallback)
def show_frame(frame):
    try:
        # Resize the frame to a smaller size for better visibility
        resized_frame = cv2.resize(frame, (800, 600))  # Resize to 800x600 resolution
        cv2.imshow('Vehicle Detection', resized_frame)
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        # Fallback to matplotlib if cv2.imshow fails
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Vehicle Detection')
        plt.axis('off')  # Hide axes
        plt.show()

# Main program loop
cap = cv2.VideoCapture(r'C:\Users\Lenovo\Desktop\traffic_video_modified.mp4')  # Open the video file
frame_counter = 0  # Initialize a frame counter

while True:
    ret, frame = cap.read()  # Capture a frame
    if not ret or frame is None or frame.size == 0:  # Check if the frame is invalid
        print("Invalid frame detected!")
        continue  # Skip to the next iteration of the loop

    frame_counter += 1  # Increment the frame counter

    # Process every 5th frame
    if frame_counter % 5 == 0:
        detect_and_recognize(frame)  # Process the valid frame
        show_frame(frame)  # Show the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
conn.close()  # Close the database connection