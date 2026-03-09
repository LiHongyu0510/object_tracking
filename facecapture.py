import cv2
import time
import os
from datetime import datetime

class FaceCaptureSystem:
    def __init__(self, cascade_path=None, save_dir="captured_faces", capture_interval=2.0):
        """
        初始化人脸抓拍系统
        :param cascade_path: Haar Cascade分类器路径（默认使用OpenCV内置的人脸检测器）
        :param save_dir: 保存抓拍图片的文件夹
        :param capture_interval: 抓拍最小间隔（秒），防止同一人连续抓拍过多
        """
        # 创建保存目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 加载人脸检测器
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError("无法加载人脸检测器，请检查文件路径")

        self.capture_interval = capture_interval
        self.last_capture_time = 0  # 上次抓拍时间戳

    def start(self):
        """启动摄像头并开始抓拍"""
        cap = cv2.VideoCapture(0)  # 0表示默认摄像头
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("人脸抓拍系统已启动，按'q'退出，按's'手动抓拍一张")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取视频帧")
                break

            # 水平翻转，使画面更自然（可选）
            # frame = cv2.flip(frame, 1)

            # 转换为灰度图（Haar检测需要）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)  # 最小人脸尺寸，避免检测到小噪点
            )

            # 当前时间
            current_time = time.time()

            # 遍历检测到的人脸
            for (x, y, w, h) in faces:
                # 绘制矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 抓拍条件：有足够的时间间隔
                if current_time - self.last_capture_time > self.capture_interval:
                    self._save_face(frame, x, y, w, h)
                    self.last_capture_time = current_time

            # 显示画面
            cv2.imshow('Face Capture System', frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # 手动抓拍当前帧
                self._manual_capture(frame)

        cap.release()
        cv2.destroyAllWindows()
        print("系统已退出")

    def _save_face(self, frame, x, y, w, h):
        """保存人脸区域图像"""
        # 提取人脸区域（可适当扩大范围）
        face_img = frame[y:y + h, x:x + w]

        # 生成文件名：时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        filename = os.path.join(self.save_dir, f"face_{timestamp}.jpg")

        # 保存图片
        cv2.imwrite(filename, face_img)
        print(f"[抓拍] 已保存: {filename}")

    def _manual_capture(self, frame):
        """手动保存当前完整画面（或也可检测人脸后保存）"""
        # 简单实现：保存当前整个画面
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = os.path.join(self.save_dir, f"manual_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[手动] 已保存: {filename}")

if __name__ == "__main__":
    # 创建系统实例并启动
    system = FaceCaptureSystem(
        save_dir="my_faces",
        capture_interval=3.0  # 抓拍间隔3秒
    )
    system.start()