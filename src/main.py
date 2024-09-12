import cv2
import argparse
from detector.py import ObjectDetector
from tracker.py import ObjectTracker

def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    detector = ObjectDetector(
        config_path='data/yolov3.cfg',
        weights_path='data/yolov3.weights',
        classes_path='data/coco.names'
    )
    tracker = ObjectTracker()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objects = detector.detect_objects(frame)
        if objects:
            detector.draw_boxes(frame, objects)

        # (Optional) Add tracking to the detected objects
        for obj in objects:
            tracker.add_tracker(frame, obj['box'])

        success, boxes = tracker.update(frame)
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('Object Detection and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to input video file', required=True)
    parser.add_argument('--output', help='Path to output video file', required=True)
    args = parser.parse_args()
    main(args.input, args.output)
