import cv2

class ObjectTracker:
    def __init__(self):
        self.trackers = cv2.MultiTracker_create()

    def add_tracker(self, frame, box):
        tracker = cv2.TrackerCSRT_create()
        self.trackers.add(tracker, frame, tuple(box))

    def update(self, frame):
        success, boxes = self.trackers.update(frame)
        return success, boxes
