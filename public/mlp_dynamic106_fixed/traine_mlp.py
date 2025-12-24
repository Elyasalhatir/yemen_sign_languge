# -*- coding: utf-8 -*-
"""
MLP-Trainer (106-D: 82 static + 24 dynamic stats)
- video-level split
- scaler fit on train only
- evaluate both vs single hand
Saves to: /content/drive/MyDrive/final_project/models/mlp_dynamic106_fixed
"""
import json, numpy as np, pathlib, joblib, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

# -----------------------------------------------------------
JSON_PATH = pathlib.Path("/content/drive/MyDrive/final_project/all_keypoints3.json")
SAVE_DIR  = pathlib.Path("/content/drive/MyDrive/final_project/models/mlp_dynamic106_fixed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

word_hand_choice = {
    'ANCLE': 'both', 'BROTHER': 'left', 'EASY': 'left', 'ENGAGEMENT': 'left',
    'FAMILY': 'both', 'FATHER': 'left', 'HIM': 'left', 'HOUR': 'both',
    'HOW': 'left', 'MINE': 'left', 'MOTHER': 'left', 'MOUNTH': 'left',
    'NAME': 'left', 'NO': 'left', 'PERCENTAGE': 'left', 'RAEDY': 'both',
    'WHAT': 'left', 'WHEN': 'left', 'WHERE': 'left', 'YES': 'left',
    'cancer': 'both', 'cold': 'right', 'eat': 'right', 'face': 'right',
    'fever': 'right', 'loss of hair': 'right', 'medicine': 'right', 'muscle': 'both'
}
# -----------------------------------------------------------

def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    if kpts is None or len(kpts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    center = np.mean(kpts, axis=0)
    centered = kpts - center
    max_dist = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    return centered / max_dist

def frame_to_feature_vector(frm: dict, side: str, prev_frm: dict = None, next_frm: dict = None) -> dict:
    hands = frm.get("hands", {})
    hand_block = hands.get(side)
    if not hand_block or "landmarks" not in hand_block:
        return None

    # 1) static 82-D
    lm = hand_block["landmarks"]
    pts = np.array([[lm[str(i)]["x"], lm[str(i)]["y"], lm[str(i)]["z"]]
                    for i in range(21)], dtype=np.float32)
    hand_vec = normalize_keypoints(pts).flatten()
    angles = [hand_block.get("angles", {}).get(f"finger_{i}", 0.0) for i in range(1, 5)]
    arm = frm.get("arm", {})
    keys = [f"{side.upper()}_SHOULDER", f"{side.upper()}_ELBOW", f"{side.upper()}_WRIST"]
    arm_pts = np.array([[arm[k]["x"], arm[k]["y"], arm[k]["z"]]
                        for k in keys if k in arm], dtype=np.float32)
    if len(arm_pts) != 3:
        arm_pts = np.zeros((3, 3), dtype=np.float32)
    arm_vec = normalize_keypoints(arm_pts).flatten()
    nose = arm.get("NOSE")
    wrist = pts[0]
    if nose:
        nose_pt = np.array([nose["x"], nose["y"], nose["z"]], dtype=np.float32)
        dist_vec = wrist - nose_pt
    else:
        dist_vec = np.zeros(3, dtype=np.float32)
    static = np.concatenate([hand_vec, angles, arm_vec, dist_vec])

    # 2) dynamic 6-D
    def get_wrist(frm_dict):
        if not frm_dict: return None
        h = frm_dict.get("hands", {}).get(side, {})
        if "landmarks" not in h: return None
        return np.array([h["landmarks"]["0"]["x"], h["landmarks"]["0"]["y"], h["landmarks"]["0"]["z"]], dtype=np.float32)

    w_prev = get_wrist(prev_frm)
    w_curr = wrist
    w_next = get_wrist(next_frm)

    vel = np.zeros(3)
    acc = np.zeros(3)
    if w_prev is not None and w_curr is not None:
        vel = (w_curr - w_prev) * 30  # FPS=30
    if all(v is not None for v in [w_prev, w_curr, w_next]):
        vel_prev = (w_curr - w_prev) * 30
        vel_next = (w_next - w_curr) * 30
        acc = (vel_next - vel_prev) * 30
    dynamic = np.concatenate([vel, acc])
    return {"static": static, "dynamic": dynamic}

def load_data():
    print("📖 Loading all_keypoints.json ...")
    data = json.load(open(JSON_PATH, encoding="utf-8"))
    word_to_idx = {w: i for i, w in enumerate(sorted({v["word"] for v in data}))}

    video_features, video_labels, video_hand_type = [], [], []
    for vid in tqdm(data, desc="videos"):
        word = vid["word"]
        label = word_to_idx[word]
        choice = word_hand_choice.get(word, 'both')
        sides = ['left', 'right'] if choice == 'both' else [choice]

        frames = vid["frames"]
        statics, dynamics = [], []
        for i, frm in enumerate(frames):
            prev = frames[i-1] if i > 0 else None
            nxt  = frames[i+1] if i < len(frames)-1 else None
            for side in sides:
                feat = frame_to_feature_vector(frm, side, prev, nxt)
                if feat is not None:
                    statics.append(feat["static"])
                    dynamics.append(feat["dynamic"])

        if statics:
            static_mean = np.mean(statics, axis=0)  # 82
            dyn_stats = np.concatenate([
                np.mean(dynamics, axis=0),  # 6
                np.std (dynamics, axis=0),  # 6
                np.min (dynamics, axis=0),  # 6
                np.max (dynamics, axis=0)   # 6
            ])  # 24
            video_features.append(np.concatenate([static_mean, dyn_stats]))  # 106
            video_labels.append(label)
            video_hand_type.append(choice)

    X = np.array(video_features, dtype=np.float32)
    y = np.array(video_labels, dtype=np.int32)
    print(f"✅ Video-level data:  X = {X.shape}  (static 82 + dynamic 24)")
    return X, y, video_hand_type, word_to_idx

def evaluate_subset(y_true, y_pred, idx_to_word, title):
    """
    y_true : array of integers (class indices)
    y_pred : array of integers (class indices)
    idx_to_word : dict mapping index -> word
    title  : str
    """
    print(f"\n----- {title} -----")
    labels = [idx_to_word[i] for i in sorted(set(y_true))]
    print(classification_report(y_true, y_pred,
                                target_names=labels,
                                digits=3, zero_division=0))
    print("Per-class accuracy:")
    for lab_idx in sorted(set(y_true)):
        mask = (y_true == lab_idx)
        print(f"{idx_to_word[lab_idx]}: {accuracy_score(y_true[mask], y_pred[mask]):.3f}")
def main():
    X, y, hand_type, word_to_idx = load_data()
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    # video-level split
    X_train, X_test, y_train, y_test, h_train, h_test = train_test_split(
        X, y, hand_type, test_size=0.2, random_state=42, stratify=y)

    # scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # train
    print("\n🎓 Training MLP ...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)

    # evaluate
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Overall Accuracy: {acc:.1%}")
    labels = [idx_to_word[i] for i in sorted(idx_to_word)]
    print(classification_report(y_test, y_pred, target_names=labels, digits=3, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("MLP Confusion Matrix (video-level, 106-D)")
    plt.tight_layout()
    plt.show()

    # both vs single
    both_mask   = [h == 'both' for h in h_test]
    single_mask = [h != 'both' for h in h_test]

    if np.any(both_mask):
        evaluate_subset(y_test[both_mask], y_pred[both_mask],
                        idx_to_word,
                        "Words that use BOTH hands")
    if np.any(single_mask):
        evaluate_subset(y_test[single_mask], y_pred[single_mask],
                        idx_to_word,
                        "Words that use SINGLE hand")
    # save
    joblib.dump(mlp, SAVE_DIR / "mlp_model.pkl")
    joblib.dump(scaler, SAVE_DIR / "\scaler.pkl")
    pickle.dump(word_to_idx, open(SAVE_DIR / "word_to_idx.pkl", "wb"))
    pickle.dump(idx_to_word, open(SAVE_DIR / "idx_to_word.pkl", "wb"))
    print("\n💾 Models saved to:", SAVE_DIR)

if __name__ == "__main__":
    main()