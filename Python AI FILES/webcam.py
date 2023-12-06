import cv2

# Replace with DVR's IP address, username, password, and channel number
#dvr_ip = "192.168.1.101"
#username = "admin"
#password = "cms12345"
#channel = 1

# Initialize camera
cap = cv2.VideoCapture(0)  # You can change the parameter to the camera index if you have multiple cameras

# Set up variables for loitering detection
loitering_time_threshold = 180  # 3 minutes in seconds
start_time = None
loitering = False

# Construct the RTSP URL
#rtsp_url = f"rtsp://{username}:{password}@{dvr_ip}:554/h264/ch{channel}/main/av_stream"

# Create a VideoCapture object using the RTSP URL
#cap = cv2.VideoCapture(rtsp_url)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was properly captured
    if ret:
        # Display the captured frame
        cv2.imshow('DVR Stream', frame)

        # Check if the user wants to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break if the capture is not working
    else:
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()