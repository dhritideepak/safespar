# safespar
Computer vision pipeline assessing martial arts safety from video footage using rule-based biomechanics scoring (knee flexion, elbow extension, shoulder/hip tilt, torso lean) from YOLOv8n-pose keypoints. Has filtering pipeline to clean noisy detections, plus experimental learned model that can score degraded/low confidence images rules can't score

Research Question
Can a learned model extend geometric safety scaring to cases analytical (rule-based) methods must abstain?

Pipeline
1. Pose Estimation; YOLOv8n-pose extracts keypoints from frames
2. Detection Filtering; four-stage filter removes unreliable detections:
   1.cut_ghosts uses an area threshold to filter out tiny detections
   2.filter_by_confidence uses YOLO's own keypoint confidences to cut out low-confidence detections
   3.eliminate_overlap; removes detections that heavily overlap (YOLO tends to perform badly with these cases)
   4.check_anatomy; limb-ratio check for merged detections making sure poses are anatomically plausible
3. Biomechanical Scoring;five parameters computed from keypoints;
   -Knee flexion
   -Elbow extension
   -Shoulder tilt
   -Hip tilt
   -Torso lean
    Each scored 0-100 via score_parameter(), with confidence gating per parameter. Overall score averages only over parameters that fire.

Dataset
Assembled via Roboflow: taekwondo and boxing images (~3000 boxing images, with ringside shots filtered out). Karate images were considered but dropped due to insufficient resolution.

Output
CSV files (train/valid/test) with 51 keypoint columns and 6 score columns per detected person. Missing parameters stored as NaN

Tools
  -Google Colab (GPU runtime)
  -Ultralytics YOLOv8n-pose
  -pandas, OpenCV
  -Roboflow

Status
Work in progress - science fair project. Biomechanical thresholds under validation with expert
