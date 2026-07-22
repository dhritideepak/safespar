# safespar
Computer vision pipeline assessing martial arts safety from video footage using rule-based biomechanics scoring (knee flexion, elbow extension, shoulder/hip tilt, torso lean) from YOLOv8n-pose keypoints. Has filtering pipeline to clean noisy detections, plus experimental learned model that can score degraded/low confidence images rules can't score

# Martial Arts Safety Assessment via Computer Vision

## Overview
A computer vision pipeline that assesses martial arts safety from video footage. Pose keypoints are extracted with YOLOv8n-pose, then used to compute biomechanical safety scores through a rule-based system. An experimental learned model explores whether it can extend safety scoring to cases where the rule-based method must abstain due to degraded or low-confidence images.

## Research Question
Can a learned model extend geometric safety scoring to cases where analytical (rule-based) methods must abstain?

## Pipeline
1. **Pose Estimation** — YOLOv8n-pose (Ultralytics) extracts keypoints from frames
2. **Detection Filtering** — four-stage filter chain removes unreliable detections:
   - `cut_ghosts` — area threshold filtering out tiny detections
   - `filter_by_confidence` — uses YOLO's own keypoint confidence scores to filter out low-confidence detections
   - `eliminate_overlap` — IoU-based duplicate removal - YOLO doesn't perform well in cases of high person-to-person overlap
   - `check_anatomy` — limb-ratio check for merged detections
3. **Biomechanical Scoring** — five parameters computed from keypoints:
   - Knee flexion
   - Elbow extension
   - Shoulder tilt
   - Hip tilt
   - Torso lean

   Each scored 0–100 via `score_parameter()`, with confidence gating per parameter. Overall score averages only over parameters that fire.

## Dataset
Assembled via Roboflow: taekwondo and boxing images (~3,000 boxing images, filtered for ringside shots). Karate was dropped due to insufficient resolution. Train/valid/test splits preserved from Roboflow to prevent augmentation leakage.

## Tools
- Google Colab (GPU runtime)
- Ultralytics YOLOv8n-pose
- pandas, OpenCV
- Roboflow

## Status
Work in progress — science fair project. Biomechanical thresholds under validation with a sports medicine consultant.
