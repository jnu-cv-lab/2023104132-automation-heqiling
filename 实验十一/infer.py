import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import os
from train import SkeletonTransformer  # 复用训练代码的模型类

# ===================== 参数配置 =====================
TARGET_FRAME = 30
FRAME_DIM = 132
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_output/badminton_transformer.pth"
LABEL_PATH = "skeleton_npy/label_map.json"
DEMO_VIDEO = "demo.mp4"

# ===================== MediaPipe 初始化（与预处理一致） =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== 视频转骨架序列（与预处理逻辑完全一致） =====================
def video_to_skeleton_seq(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{vid_path}")
    
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        frame_feat = np.zeros(FRAME_DIM, dtype=np.float32)
        if res.pose_landmarks:
            for idx, lm in enumerate(res.pose_landmarks.landmark):
                off = idx * 4
                frame_feat[off] = lm.x
                frame_feat[off+1] = lm.y
                frame_feat[off+2] = lm.z
                frame_feat[off+3] = lm.visibility
        frame_list.append(frame_feat)
    cap.release()

    # 重采样
    orig_len = len(frame_list)
    sample_idx = np.linspace(0, orig_len-1, TARGET_FRAME, dtype=int)
    sample_frames = [frame_list[i] for i in sample_idx]
    seq = np.stack(sample_frames, axis=0)

    # 归一化（与预处理完全一致）
    left_hip_x = seq[:, 23*4]
    left_hip_y = seq[:, 23*4+1]
    right_hip_x = seq[:, 24*4]
    right_hip_y = seq[:, 24*4+1]
    center_x = (left_hip_x + right_hip_x) / 2
    center_y = (left_hip_y + right_hip_y) / 2

    shoulder_x1 = seq[:, 11*4]
    shoulder_x2 = seq[:, 12*4]
    scale = np.abs(shoulder_x1 - shoulder_x2)
    scale[scale < 1e-6] = 1.0

    for i in range(33):
        off = i * 4
        seq[:, off] = (seq[:, off] - center_x) / scale
        seq[:, off+1] = (seq[:, off+1] - center_y) / scale

    return seq

# ===================== 主推理流程 =====================
if __name__ == "__main__":
    # 1. 提取骨架序列
    print(f"正在处理视频：{DEMO_VIDEO}")
    seq_data = video_to_skeleton_seq(DEMO_VIDEO)

    # 2. 转为模型输入张量
    input_tensor = torch.from_numpy(seq_data).float().unsqueeze(0).to(DEVICE)

    # 3. 加载模型
    model = SkeletonTransformer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 4. 推理预测
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    # 5. 加载标签映射
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    pred_class_name = label_map[str(pred_idx)]

    # 6. 输出结果
    print("\n" + "="*40)
    print(f"Predicted class: {pred_class_name}")
    print(f"Confidence: {confidence:.2f}")
    print("="*40)