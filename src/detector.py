
The YOLO object detection logic, which loads the model and performs detection.

```python
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, config_path, weights_path, classes_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        objects = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            objects.append({
                'class_id': class_ids[i],
                'confidence': confidences[i],
                'box': box,
                'color': self.colors[class_ids[i]]
            })
        return objects

    def draw_boxes(self, image, objects):
        for obj in objects:
            x, y, w, h = obj['box']
            class_id = obj['class_id']
            confidence = obj['confidence']
            color = obj['color']

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image
