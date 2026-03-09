import cv2

class ObjectTracker:

    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()
        self.initialized = False

    def init_tracker(self, frame, bbox):
        """
        初始化跟踪器
        """
        self.tracker.init(frame, bbox)
        self.initialized = True

    def update(self, frame):
        """
        更新目标位置
        """
        success, bbox = self.tracker.update(frame)
        return success, bbox