 Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
📖 Loading all_keypoints.json ...
videos: 100%|██████████| 3287/3287 [00:24<00:00, 136.91it/s]
✅ Video-level data:  X = (3246, 103)  (static 82 + dynamic 24)

🎓 Training MLP ...

✅ Overall Accuracy: 99.7%
              precision    recall  f1-score   support

       ANCLE      1.000     1.000     1.000        23
     BROTHER      1.000     1.000     1.000        16
        EASY      1.000     1.000     1.000        21
  ENGAGEMENT      0.955     1.000     0.977        21
      FAMILY      1.000     1.000     1.000        20
      FATHER      0.933     1.000     0.966        14
         HIM      1.000     1.000     1.000        21
        HOUR      1.000     1.000     1.000        25
         HOW      1.000     1.000     1.000        21
        MINE      1.000     1.000     1.000        21
      MOTHER      1.000     0.929     0.963        14
      MOUNTH      1.000     1.000     1.000        21
        NAME      1.000     1.000     1.000        21
          NO      1.000     1.000     1.000        21
  PERCENTAGE      1.000     1.000     1.000        21
       RAEDY      1.000     1.000     1.000        21
        WHAT      1.000     0.952     0.976        21
        WHEN      1.000     1.000     1.000        21
       WHERE      1.000     1.000     1.000        21
         YES      1.000     1.000     1.000        21
      cancer      1.000     1.000     1.000        31
        cold      1.000     1.000     1.000        30
         eat      1.000     1.000     1.000        31
        face      1.000     1.000     1.000        30
       fever      1.000     1.000     1.000        30
loss of hair      1.000     1.000     1.000        31
    medicine      1.000     1.000     1.000        31
      muscle      1.000     1.000     1.000        30

    accuracy                          0.997       650
   macro avg      0.996     0.996     0.996       650
weighted avg      0.997     0.997     0.997       650


----- Words that use BOTH hands -----
              precision    recall  f1-score   support

       ANCLE      1.000     1.000     1.000        23
      FAMILY      1.000     1.000     1.000        20
        HOUR      1.000     1.000     1.000        25
       RAEDY      1.000     1.000     1.000        21
      cancer      1.000     1.000     1.000        31
      muscle      1.000     1.000     1.000        30

    accuracy                          1.000       150
   macro avg      1.000     1.000     1.000       150
weighted avg      1.000     1.000     1.000       150

Per-class accuracy:
ANCLE: 1.000
FAMILY: 1.000
HOUR: 1.000
RAEDY: 1.000
cancer: 1.000
muscle: 1.000

----- Words that use SINGLE hand -----
              precision    recall  f1-score   support

     BROTHER      1.000     1.000     1.000        16
        EASY      1.000     1.000     1.000        21
  ENGAGEMENT      0.955     1.000     0.977        21
      FATHER      0.933     1.000     0.966        14
         HIM      1.000     1.000     1.000        21
         HOW      1.000     1.000     1.000        21
        MINE      1.000     1.000     1.000        21
      MOTHER      1.000     0.929     0.963        14
      MOUNTH      1.000     1.000     1.000        21
        NAME      1.000     1.000     1.000        21
          NO      1.000     1.000     1.000        21
  PERCENTAGE      1.000     1.000     1.000        21
        WHAT      1.000     0.952     0.976        21
        WHEN      1.000     1.000     1.000        21
       WHERE      1.000     1.000     1.000        21
         YES      1.000     1.000     1.000        21
        cold      1.000     1.000     1.000        30
         eat      1.000     1.000     1.000        31
        face      1.000     1.000     1.000        30
       fever      1.000     1.000     1.000        30
loss of hair      1.000     1.000     1.000        31
    medicine      1.000     1.000     1.000        31

    accuracy                          0.996       500
   macro avg      0.995     0.995     0.995       500
weighted avg      0.996     0.996     0.996       500

Per-class accuracy:
BROTHER: 1.000
EASY: 1.000
ENGAGEMENT: 1.000
FATHER: 1.000
HIM: 1.000
HOW: 1.000
MINE: 1.000
MOTHER: 0.929
MOUNTH: 1.000
NAME: 1.000
NO: 1.000
PERCENTAGE: 1.000
WHAT: 0.952
WHEN: 1.000
WHERE: 1.000
YES: 1.000
cold: 1.000
eat: 1.000
face: 1.000
fever: 1.000
loss of hair: 1.000
medicine: 1.000

💾 Models saved to: /content/drive/MyDrive/final_project/models/mlp_dynamic106_fixed

