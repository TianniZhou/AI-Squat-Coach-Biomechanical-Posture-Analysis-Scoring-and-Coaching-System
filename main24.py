import cv2
import mediapipe as mp
import math
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0
    cos_val = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_val))

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_avg(lm, w, h, left_name, right_name):
    left = getattr(mp_pose.PoseLandmark, f"LEFT_{left_name}").value
    right = getattr(mp_pose.PoseLandmark, f"RIGHT_{right_name}").value
    avg_x = (lm[left].x + lm[right].x) / 2
    avg_y = (lm[left].y + lm[right].y) / 2
    return (int(avg_x * w), int(avg_y * h))

# ============================================================================
# STANDARD 1: Squat Depth (25%)
# ============================================================================

def score_depth(knee_angle):
    if knee_angle >= 150:
        return 0
    elif knee_angle >= 50:
        return (150 - knee_angle) / 100 * 100
    else:
        return knee_angle / 50 * 100

# ============================================================================
# STANDARD 2: Core Stability (15%)
# ============================================================================

def score_core(shoulder, hip, knee, ankle, leg_len, facing_right):
    if facing_right:
        cop = ankle[0] + leg_len * 0.35
    else:
        cop = ankle[0] - leg_len * 0.35
    
    com = (shoulder[0] * 0.55 + hip[0] * 0.25 + knee[0] * 0.145 + ankle[0] * 0.055)
    
    foot_len = leg_len * 0.65
    offset = abs(com - cop) / foot_len if foot_len > 0 else 1
    
    return max(0, 100 - offset * 40)

# ============================================================================
# STANDARD 3: Joint Coupling (20%)
# ============================================================================

def score_coupling(knee_angle, torso_angle):
    if knee_angle < 70:
        low, high = 30, 40
    elif knee_angle < 100:
        r = (knee_angle - 70) / 30
        low, high = 30 - r * 15, 40 - r * 15
    else:
        low, high = 5, 15
    
    if low <= torso_angle <= high:
        deviation = 0
    else:
        deviation = min(abs(torso_angle - low), abs(torso_angle - high))
    
    return max(0, 100 - deviation * (100 / 30))

# ============================================================================
# STANDARD 4: Shoulder Position (15%)
# Penalizes shoulder ahead of knee
# ============================================================================

def score_shoulder(shoulder, knee, leg_len, facing_right):
    ideal = leg_len * 0.1
    
    if facing_right:
        if shoulder[0] <= knee[0]:
            behind = knee[0] - shoulder[0]
            if behind >= ideal:
                score = 100
            else:
                score = 70 + (behind / ideal) * 30
        else:
            ahead = shoulder[0] - knee[0]
            score = max(0, 70 - (ahead / leg_len) * 700)
    else:
        if shoulder[0] >= knee[0]:
            behind = shoulder[0] - knee[0]
            if behind >= ideal:
                score = 100
            else:
                score = 70 + (behind / ideal) * 30
        else:
            ahead = knee[0] - shoulder[0]
            score = max(0, 70 - (ahead / leg_len) * 700)
    
    return min(100, max(0, score))

# ============================================================================
# STANDARD 5: Moment Ratio (10%)
# REVISED: More realistic scoring for normal squat postures
# Normal range: 0.6-1.2, Elite: >1.2
# ============================================================================

def score_moment(hip, knee, ankle, facing_right):
    # Gravity line using full body COM for better accuracy
    gravity = (hip[0] * 0.4 + knee[0] * 0.35 + ankle[0] * 0.25)
    
    if facing_right:
        hip_lever = abs(gravity - hip[0])
        knee_lever = abs(gravity - knee[0])
    else:
        hip_lever = abs(hip[0] - gravity)
        knee_lever = abs(knee[0] - gravity)
    
    if knee_lever == 0:
        return 50
    
    ratio = hip_lever / knee_lever
    
    # Revised realistic thresholds:
    # ratio > 1.0 = hip dominant (excellent)
    # ratio 0.7-1.0 = balanced (good)
    # ratio 0.5-0.7 = slight knee dominant (fair)
    # ratio < 0.5 = knee dominant (poor)
    if ratio >= 1.0:
        return 100
    elif ratio >= 0.7:
        return 70 + (ratio - 0.7) / 0.3 * 30  # 0.7->70, 1.0->100
    elif ratio >= 0.5:
        return 40 + (ratio - 0.5) / 0.2 * 30  # 0.5->40, 0.7->70
    else:
        return ratio / 0.5 * 40  # 0->0, 0.5->40

# ============================================================================
# STANDARD 6: Knee Position (15%)
# 0-0.4: Full points (was 0.3)
# 0.4-0.8: Gentle linear penalty (was 0.3-0.6 severe)
# >0.8: Exponential penalty
# ============================================================================

def score_knee_pos(knee, ankle, leg_len, facing_right, knee_angle):
    foot_len = leg_len * 0.65
    
    if facing_right:
        over = max(0, knee[0] - ankle[0])
    else:
        over = max(0, ankle[0] - knee[0])
    
    ratio = over / foot_len if foot_len > 0 else 0
    
    # Depth adjustment: deeper squats naturally have more knee over toe
    # More lenient for deep squats (knee_angle < 90)
    depth_factor = 1.0
    if knee_angle < 90:
        # Deep squat: 20% more tolerance
        depth_factor = 1.2
    elif knee_angle > 120:
        # Shallow squat: 20% less tolerance
        depth_factor = 0.8
    
    adjusted_ratio = ratio / depth_factor
    
    # Thresholds
    if adjusted_ratio <= 0.4:
        return 100
    elif adjusted_ratio <= 0.8:
        # Gentle linear: 0.4->100, 0.8->60
        return 100 - (adjusted_ratio - 0.4) / 0.4 * 40
    else:
        # Exponential beyond 0.8
        return 60 * math.exp(-(adjusted_ratio - 0.8) / 0.3)

# ============================================================================
# MAIN PROGRAM
# ============================================================================

filename = input("Enter image filename: ")
image = cv2.imread(filename)
if image is None:
    print("Image not found")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print("No person detected")
        exit()
    
    lm = results.pose_landmarks.landmark
    h, w, _ = image.shape
    
    shoulder = get_avg(lm, w, h, "SHOULDER", "SHOULDER")
    hip = get_avg(lm, w, h, "HIP", "HIP")
    knee = get_avg(lm, w, h, "KNEE", "KNEE")
    ankle = get_avg(lm, w, h, "ANKLE", "ANKLE")
    
    knee_angle = safe_angle(hip, knee, ankle)
    leg_len = distance(knee, ankle)
    
    dx = abs(shoulder[0] - hip[0])
    dy = abs(shoulder[1] - hip[1])
    torso_angle = math.degrees(math.atan2(dx, dy)) if dy > 0 else 0
    
    facing_right = knee[0] > hip[0]
    
    print(f"\n{'='*50}")
    print(f"Facing: {'RIGHT' if facing_right else 'LEFT'}")
    print(f"Knee angle: {int(knee_angle)}° | Torso angle: {int(torso_angle)}°")
    
    # Gate: Depth check
    if knee_angle >= 150:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(image, "FAILED: Knee angle > 150 (not a squat)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 150, 200), 2)
        i = 1
        while os.path.exists(f"output_{i}.jpg"):
            i += 1
        cv2.imwrite(f"output_{i}.jpg", image)
        print(f"Failed - saved output_{i}.jpg")
        exit()
    
    # Calculate all 6 scores
    s1_depth = score_depth(knee_angle)
    s2_core = score_core(shoulder, hip, knee, ankle, leg_len, facing_right)
    s3_coupling = score_coupling(knee_angle, torso_angle)
    s4_shoulder = score_shoulder(shoulder, knee, leg_len, facing_right)
    s5_moment = score_moment(hip, knee, ankle, facing_right)
    s6_knee = score_knee_pos(knee, ankle, leg_len, facing_right, knee_angle)
    
    total = int(s1_depth * 0.25 + s2_core * 0.15 + s3_coupling * 0.20 + 
                s4_shoulder * 0.15 + s5_moment * 0.10 + s6_knee * 0.15)
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Colors (BGR)
    PINK = (180, 150, 200)
    GREEN = (110, 160, 130)
    BLUE = (160, 140, 100)
    YELLOW = (100, 190, 190)
    ORANGE = (120, 140, 210)
    PURPLE = (180, 120, 160)
    RED = (100, 100, 200)
    
    # Display scores
    cv2.putText(image, f"TOTAL: {total}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, PINK, 2)
    cv2.putText(image, f"1.Depth(25%): {int(s1_depth)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)
    cv2.putText(image, f"2.Core(15%): {int(s2_core)}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLUE, 2)
    cv2.putText(image, f"3.Coupling(20%): {int(s3_coupling)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
    cv2.putText(image, f"4.Shoulder(15%): {int(s4_shoulder)}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE, 2)
    cv2.putText(image, f"5.Moment(10%): {int(s5_moment)}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PURPLE, 2)
    cv2.putText(image, f"6.KneePos(15%): {int(s6_knee)}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
    
    # Warning for shoulder ahead of knee
    shoulder_ahead = (facing_right and shoulder[0] > knee[0]) or (not facing_right and shoulder[0] < knee[0])
    if shoulder_ahead:
        cv2.putText(image, "WARNING: Shoulder ahead of knee!", (50, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    
    # Console output
    print(f"\n{'='*50}")
    print(f"1.Depth(25%):     {int(s1_depth)}/100")
    print(f"2.Core(15%):      {int(s2_core)}/100")
    print(f"3.Coupling(20%):  {int(s3_coupling)}/100")
    print(f"4.Shoulder(15%):  {int(s4_shoulder)}/100")
    print(f"5.Moment(10%):    {int(s5_moment)}/100")
    print(f"6.KneePos(15%):   {int(s6_knee)}/100")
    print(f"{'-'*50}")
    print(f"TOTAL SCORE: {total}/100")
    print(f"{'='*50}")
    
    if total >= 85:
        print("EXCELLENT - Elite level technique")
    elif total >= 70:
        print("GOOD - Solid form, low injury risk")
    elif total >= 55:
        print("FAIR - Acceptable, needs improvement")
    elif total >= 40:
        print("POOR - High injury risk, seek coaching")
    else:
        print("CRITICAL - Stop lifting, major form issues")
    
    i = 1
    while os.path.exists(f"output_{i}.jpg"):
        i += 1
    cv2.imwrite(f"output_{i}.jpg", image)
    print(f"\nSaved output_{i}.jpg")