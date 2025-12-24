"""
Keypoints + Advanced Arm-Hand Features (JSON per class)
يتم تجاهل أي إطار لا يحتوي على يد
دمج كل الملفات في ملف واحد + رفعها إلى Google Drive
"""
# ------------------  استيراد المكتبات  ------------------
import cv2, mediapipe as mp, json, os, shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from natsort import natsorted

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

FPS = 30
ARM_LANDMARKS = {
    'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW.value,
    'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST.value,
    'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST.value,
    'NOSE': mp_pose.PoseLandmark.NOSE.value,
}
# -------------------------------------------------------------------

class ArmHandKeypointExtractor:
    def __init__(self, output_dir='keypoints_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # ✅ دالة آمنة للحصول على الإحداثيات
    def _get_landmark_xyz(self, landmarks, idx):
        if landmarks is None:
            return None
        lm = landmarks.landmark[idx]
        return {'x': lm.x, 'y': lm.y, 'z': lm.z}

    # ✅ دالة velocity بدون استخدام arrays في if
    def velocity(self, p1, p2):
        if not isinstance(p1, dict) or not isinstance(p2, dict):
            return {'x': 0, 'y': 0, 'z': 0}
        return {
            'x': (p2['x'] - p1['x']) * FPS,
            'y': (p2['y'] - p1['y']) * FPS,
            'z': (p2['z'] - p1['z']) * FPS
        }

    # ✅ دالة acceleration بدون استخدام arrays في if
    def acceleration(self, p_prev, p_curr, p_next):
        if not all(isinstance(p, dict) for p in [p_prev, p_curr, p_next]):
            return {'x': 0, 'y': 0, 'z': 0}
        v1 = self.velocity(p_prev, p_curr)
        v2 = self.velocity(p_curr, p_next)
        dt = 1.0 / FPS
        return {
            'x': (v2['x'] - v1['x']) / dt,
            'y': (v2['y'] - v1['y']) / dt,
            'z': (v2['z'] - v1['z']) / dt
        }

    # ✅ حساب زوايا الأصابع
    def finger_angles(self, hand_landmarks):
        angles = {}
        if not hand_landmarks:
            return angles
        lm = hand_landmarks.landmark
        tips, dips, mcps = [4, 8, 12, 16, 20], [3, 7, 11, 15, 19], [2, 5, 9, 13, 17]
        for idx in range(1, 5):
            try:
                a = np.array([lm[mcps[idx]].x, lm[mcps[idx]].y, lm[mcps[idx]].z])
                b = np.array([lm[dips[idx]].x, lm[dips[idx]].y, lm[dips[idx]].z])
                c = np.array([lm[tips[idx]].x, lm[tips[idx]].y, lm[tips[idx]].z])
                cosine = np.dot(b - a, c - b) / (np.linalg.norm(b - a) * np.linalg.norm(c - b) + 1e-6)
                angles[f'finger_{idx}'] = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            except:
                angles[f'finger_{idx}'] = 0.0
        return angles

    # ✅ استخراج معالم الإطار الواحد + تجاهل الإطار إذا لم تُكتشف يد
    def extract_frame(self, frame, prev_frame_data, next_frame_data):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb)
        hands_res = self.hands.process(rgb)

        # إذا لم تُكتشف أي يد، نعيد None
        if not hands_res or not hands_res.multi_hand_landmarks:
            return None

        arm = {}
        if pose_res and pose_res.pose_landmarks:
            arm = {name: self._get_landmark_xyz(pose_res.pose_landmarks, idx) for name, idx in ARM_LANDMARKS.items()}

        hands_out = {'left': {}, 'right': {}}
        for h_idx, h_landmarks in enumerate(hands_res.multi_hand_landmarks):
            label = hands_res.multi_handedness[h_idx].classification[0].label.lower()
            hand_points = {i: self._get_landmark_xyz(h_landmarks, i) for i in range(21)}
            hands_out[label] = {
                'landmarks': hand_points,
                'angles': self.finger_angles(h_landmarks)
            }

        nose = arm.get('NOSE')
        for side in ['left', 'right']:
            wrist = hands_out[side].get('landmarks', {}).get(0)
            if isinstance(wrist, dict) and isinstance(nose, dict):
                dx = wrist['x'] - nose['x']
                dy = wrist['y'] - nose['y']
                dz = wrist['z'] - nose['z']
                hands_out[side]['distance_to_nose'] = float(np.sqrt(dx*dx + dy*dy + dz*dz))

            curr = hands_out[side].get('landmarks', {}).get(0)
            prev = (prev_frame_data or {}).get('hands', {}).get(side, {}).get('landmarks', {}).get(0)
            nxt = (next_frame_data or {}).get('hands', {}).get(side, {}).get('landmarks', {}).get(0)

            if isinstance(curr, dict) and isinstance(prev, dict):
                hands_out[side]['velocity'] = self.velocity(prev, curr)
            if all(isinstance(p, dict) for p in [prev, curr, nxt]):
                hands_out[side]['acceleration'] = self.acceleration(prev, curr, nxt)

        return {'arm': arm, 'hands': hands_out}

    # ✅ معالجة الفيديو كامل + استبعاد الإطارات التي لا تحتوي على يد
    def process_video(self, video_path, word_label):
        cap = cv2.VideoCapture(str(video_path))
        all_frames, frames_data = [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        frame_count = len(all_frames)
        for i in range(frame_count):
            prev = all_frames[i-1] if i > 0 else None
            nxt = all_frames[i+1] if i < frame_count-1 else None
            prev_data = self.extract_frame(prev, None, None) if prev is not None else None
            next_data = self.extract_frame(nxt, None, None) if nxt is not None else None
            frame_data = self.extract_frame(all_frames[i], prev_data, next_data)
            if frame_data is not None:  # ✅ فقط الإطارات التي تحتوي على يد
                frames_data.append(frame_data)

        return {
            'word': word_label,
            'video_path': str(video_path),
            'frame_count': len(frames_data),  # عدد الإطارات الفعلية بعد التصفية
            'frames': frames_data
        }

    # ✅ معالجة كل الفيديوهات في مجلد الكلاسات
    def process_all_videos(self, videos_dir):
        videos_dir = Path(videos_dir)
        all_videos = []
        for word_folder in natsorted(videos_dir.iterdir()):
            if not word_folder.is_dir():
                continue
            word_label = word_folder.name
            print(f'\nProcessing class: {word_label}')
            video_files = natsorted(
                list(word_folder.glob('*.mp4')) +
                list(word_folder.glob('*.avi')) +
                list(word_folder.glob('*.mov'))
            )
            word_data = []
            for vid in tqdm(video_files, desc=word_label):
                try:
                    result = self.process_video(vid, word_label)
                    if result['frames']:  # ✅ فقط الفيديوهات التي تحتوي على إطارات صالحة
                        word_data.append(result)
                except Exception as e:
                    print(f'Error {vid}: {e}')
            out_class_dir = self.output_dir / word_label
            out_class_dir.mkdir(exist_ok=True)
            with open(out_class_dir / 'keypoints.json', 'w', encoding='utf-8') as fj:
                json.dump(word_data, fj, ensure_ascii=False, indent=2)
            print(f'Saved -> {out_class_dir}/keypoints.json  ({len(word_data)} videos)')
            all_videos.extend(word_data)
        return all_videos

    # ✅ دمج كل ملفات JSON في ملف واحد
    def merge_all_classes(self, all_videos_list, merged_name='all_keypoints.json'):
        merged_path = self.output_dir / merged_name
        with open(merged_path, 'w', encoding='utf-8') as fm:
            json.dump(all_videos_list, fm, ensure_ascii=False, indent=2)
        print(f'\n✅ Merged file saved -> {merged_path}  ({len(all_videos_list)} total videos)')

    # ✅ رفع المجلد النهائي إلى Google Drive
    def upload_to_gdrive(self, gdrive_folder='final_project'):
        from google.colab import drive
        drive.mount('/content/drive')
        dst = Path(f'/content/drive/MyDrive/{gdrive_folder}')
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.output_dir, dst)
        print(f'✅ Uploaded to Google Drive: {dst}')

    # ✅ تشغيل كامل
    def run(self, videos_dir, gdrive_folder='final_project'):
        all_vids = self.process_all_videos(videos_dir)
        self.merge_all_classes(all_vids)
        self.upload_to_gdrive(gdrive_folder)
        print('\n🎉 All done: extraction + filtering + merging + upload!')


# ================== تشغيل الكود ==================
if __name__ == '__main__':
    VIDEOS_DIR = '/content/drive/MyDrive/dataset/VIDEO DATABASE/NEW'  # غيّره حسب مسارك
    extractor = ArmHandKeypointExtractor(output_dir='keypoints_data')
    extractor.run(VIDEOS_DIR, gdrive_folder='final_project')