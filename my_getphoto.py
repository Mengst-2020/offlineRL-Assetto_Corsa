import cv2
import numpy as np


def process_video(video_path, output_path, frames_per_second=10,begin=0,end=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second)
    print(f"Video FPS: {fps}, Extracting every {frame_interval} frames.")

    frame_count = 0
    # while frame_count < int(fps*14):
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame_count += 1

    # 读取第一帧作为背景
    ret, background = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # 转换为灰度并模糊
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

    # 初始化最终叠加图像
    final_overlay = background.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count < int(fps*begin):
            continue
        if frame_count > int(fps*end):
            break
        if frame_count % frame_interval != 0:
            continue
        # 转换为灰度并模糊
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0) #(11, 11)   (21, 21)

        # 计算帧差
        diff = cv2.absdiff(background_gray, gray)
        _, threshold = cv2.threshold(diff, 6, 255, cv2.THRESH_BINARY)###############################25----50

        # 膨胀处理
        dilated = cv2.dilate(threshold, None, iterations=2)

        # 找到轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在最终叠加图像上绘制运动物体
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # 忽略小区域
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            final_overlay[y:y+h, x:x+w] = frame[y:y+h, x:x+w]  # 直接复制运动物体部分

    cap.release()

    # 保存结果
    cv2.imwrite(output_path, final_overlay)
    print(f"Result saved to {output_path}")

# import cv2
# from pathlib import Path

# def process_video(video_path, output_path, frames_per_second=10, begin=0, end=50):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, int(fps / frames_per_second))
#     print(f"Video FPS: {fps}, Extracting every {frame_interval} frames.")

#     frame_count = 0

#     # 输出目录
#     out_dir = Path(output_path)
#     out_dir.mkdir(parents=True, exist_ok=True)


#     idx = 0  # 用于输出文件名编号

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count < int(fps * begin):
#             continue
#         if frame_count > int(fps * end):
#             break
#         if frame_count % frame_interval != 0:
#             continue
#         cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)
#         idx += 1

#     cap.release()

#     print(f"Saved {idx} frames to {out_dir}")


if __name__ == "__main__":
     process_video("3.mp4", "3.png", frames_per_second=5, begin=0, end=17)
