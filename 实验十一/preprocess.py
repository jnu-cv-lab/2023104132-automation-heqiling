import cv2
import mediapipe as mp
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# ===================== 超参数配置 =====================
TARGET_FRAME = 30          # 统一视频帧数
KEYPOINT_NUM = 33          # MediaPipe 关键点数量
FEAT_PER_KP = 4            # 每个关键点特征: x,y,z,visibility
FRAME_DIM = KEYPOINT_NUM * FEAT_PER_KP  # 单帧特征维度 132
TEST_RATIO = 0.2           # 测试集比例
RAW_VIDEO_DIR = "raw_video"  # 原始视频目录
SAVE_DIR = "skeleton_npy"    # 输出npy目录

# ===================== MediaPipe 初始化（已修复参数） =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,   # 视频模式，帧间跟踪
    model_complexity=1,        # 标准精度模型
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== 类别标签映射 =====================
label_dict = {
    "forehand_drive": 0,
    "forehand_lift": 1,
    "forehand_net": 2,
    "forehand_clear": 3,
    "backhand_drive": 4,
    "backhand_net": 5
}

# ===================== 单视频骨架提取函数 =====================
def extract_video_skeleton(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"警告：无法打开视频 {vid_path}，已跳过")
        return None
    
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 色彩空间转换
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        # 初始化单帧特征
        frame_feat = np.zeros(FRAME_DIM, dtype=np.float32)
        if res.pose_landmarks:
            for idx, lm in enumerate(res.pose_landmarks.landmark):
                offset = idx * 4
                frame_feat[offset] = lm.x
                frame_feat[offset+1] = lm.y
                frame_feat[offset+2] = lm.z
                frame_feat[offset+3] = lm.visibility
        frame_list.append(frame_feat)
    cap.release()

    if len(frame_list) == 0:
        return None

    # 线性重采样到固定帧数
    orig_len = len(frame_list)
    sample_idx = np.linspace(0, orig_len-1, TARGET_FRAME, dtype=int)
    sample_frames = [frame_list[i] for i in sample_idx]
    seq = np.stack(sample_frames, axis=0)  # shape: (30, 132)

    # 骨架归一化：双髋为原点，肩宽做尺度归一
    left_hip_x = seq[:, 23*4]
    left_hip_y = seq[:, 23*4+1]
    right_hip_x = seq[:, 24*4]
    right_hip_y = seq[:, 24*4+1]
    center_x = (left_hip_x + right_hip_x) / 2
    center_y = (left_hip_y + right_hip_y) / 2

    shoulder_x1 = seq[:, 11*4]
    shoulder_x2 = seq[:, 12*4]
    scale = np.abs(shoulder_x1 - shoulder_x2)
    scale[scale < 1e-6] = 1.0  # 避免除零

    # 逐关键点归一化
    for i in range(KEYPOINT_NUM):
        off = i * 4
        seq[:, off] = (seq[:, off] - center_x) / scale
        seq[:, off+1] = (seq[:, off+1] - center_y) / scale

    return seq

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 自动创建输出目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    all_data = []
    all_label = []

    for cls_name, lab in label_dict.items():
        cls_dir = os.path.join(RAW_VIDEO_DIR, cls_name)
        if not os.path.exists(cls_dir):
            print(f"跳过不存在文件夹：{cls_name}")
            continue
        
        vid_list = [v for v in os.listdir(cls_dir) 
                    if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        print(f"处理类别 [{cls_name}]，视频数量：{len(vid_list)}")

        for vid in vid_list:
            vid_path = os.path.join(cls_dir, vid)
            try:
                seq_data = extract_video_skeleton(vid_path)
                if seq_data is not None:
                    all_data.append(seq_data)
                    all_label.append(lab)
            except Exception as e:
                print(f"视频处理失败 {vid_path}: {e}")
                continue

    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_label, dtype=np.int64)
    print(f"\n预处理完成，总有效样本数：{X.shape[0]}")
    print(f"单样本形状 (帧数, 特征维度)：{X.shape[1:]}")

    # 划分训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, shuffle=True, random_state=42, stratify=y
    )

    # 保存npy文件
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

    # 保存标签映射
    with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in label_dict.items()}, f, ensure_ascii=False, indent=2)

    print(f"训练集：{len(X_train)} 样本，测试集：{len(X_test)} 样本")
    print(f"所有数据已保存至 ./{SAVE_DIR}/ 目录")