import cv2
import math
import cvzone
from ultralytics import YOLO

# Cargar modelo YOLOv8 personalizado (se usa GPU automáticamente si está disponible)
model = YOLO("wheel.pt")

# Cargar video
cap = cv2.VideoCapture("data/laboratorio.mp4")

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100

            # Dibujar caja y confianza
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{conf}', (x1, y1 - 10), scale=1, thickness=1)

    cv2.imshow("YOLOv8 - wheel.pt (CUDA)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
