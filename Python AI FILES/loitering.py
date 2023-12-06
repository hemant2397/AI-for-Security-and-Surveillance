import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection
def detect_objects(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x, center_y, w, h = (np.array(detection[0:4]) * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return [(boxes[i], confidences[i]) for i in indices.flatten()] if indices is not None else []

# Function to check if a person is loitering
def is_loitering(previous_positions, current_position, max_distance, max_time):
    for position in previous_positions:
        distance = np.linalg.norm(np.array(position) - np.array(current_position))
        if distance < max_distance:
            return True

    return False

# Main function for loitering detection
def detect_loitering(image_path):
    cap = cv2.VideoCapture(image_path)
    previous_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objects = detect_objects(frame)

        for (box, confidence) in objects:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            current_position = [(x + w / 2, y + h / 2)]
            if is_loitering(previous_positions, current_position, max_distance=100, max_time=5):
                cv2.putText(frame, "Loitering Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Loitering Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "2023-11-29 14-07-03 (1).jpg"
    detect_loitering(image_path)
