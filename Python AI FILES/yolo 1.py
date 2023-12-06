from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8n.pt')
results = model("imp ai data/Multiple People", show=True)
cv2.waitKey(0)

