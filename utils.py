import time

class FPS:

    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def update(self):
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0
        return self.frame_count / elapsed