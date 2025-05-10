import cv2
import numpy as np

def extract_and_overlay_frames(video_path, output_path, frames_per_second=2):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        # return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # 读取第一帧作为背景（假设第一帧没有运动物体）
    ret, background = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    # 转换为灰度图像以便后续处理
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
    
    frame_count = 0
    processed_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111")
            break
            
        
        frame_count += 1
        
        # 按设定的帧率处理
        # if frame_count % frame_interval != 0:
        #     continue
        
        # 转换为灰度图像并模糊处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 计算当前帧与背景的差异
        diff = cv2.absdiff(background_gray, gray)
        _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # 膨胀处理，填充空洞
        dilated = cv2.dilate(threshold, None, iterations=2)
        
        # 找到轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 复制背景用于绘制
        overlay = background.copy()
        
        # 绘制所有检测到的运动物体
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # 忽略小区域
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            # 将当前帧中的运动物体部分覆盖到背景上
            overlay[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        
        processed_frames.append(overlay)
    
    cap.release()
    
    if not processed_frames:
        print("No frames processed.")
        return
    
    # 合并所有处理后的帧（取平均值或最后结果）
    final_output = processed_frames[-1]  # 这里取最后一帧的叠加结果
    
    # 保存结果
    cv2.imwrite(output_path, final_output)
    print(f"Result saved to {output_path}")

# 使用示例
video_path = "race_track.mp4"  # 替换为你的视频路径
output_path = "overlay_result.jpg"
extract_and_overlay_frames(video_path, output_path, frames_per_second=2)

# 使用示例
if __name__ == "__main__":
    video_path = "vedio/iql/4.webm"  # 替换为你的视频路径
    output_path = "overlay_result.jpg"
    extract_and_overlay_frames(video_path, output_path, frames_per_second=2)