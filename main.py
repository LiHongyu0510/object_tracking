import cv2
from camera import Camera
from tracker import ObjectTracker
from utils import FPS

def main():

    camera = Camera(0)
    tracker = ObjectTracker()
    fps = FPS()

    initBB = None

    while True:

        ret, frame = camera.read()
        if not ret:
            break

        if initBB is not None:

            success, box = tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        fps.update()
        current_fps = fps.get_fps()

        cv2.putText(
            frame,
            f"FPS: {current_fps:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        # 按 s 选择目标
        if key == ord("s"):

            bbox = cv2.selectROI(
                "Tracking",
                frame,
                fromCenter=False,
                showCrosshair=True
            )

            tracker.init_tracker(frame, bbox)
            initBB = bbox

        # 按 q 退出
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()