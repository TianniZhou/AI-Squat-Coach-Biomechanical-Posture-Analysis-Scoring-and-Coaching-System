import cv2
import mediapipe as mp
import math
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def safe_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0
    cos_val = dot / mag
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Standard 1: Knee Depth (40%)
def score_depth(knee_angle):
    if knee_angle >= 170:
        return 0
    elif knee_angle >= 50:
        return (170 - knee_angle) / (170 - 50) * 100
    else:
        return knee_angle / 50 * 100

# Standard 2: Core Balance (15%)
def score_core_balance(shoulder, hip, knee, ankle, foot_length, facing_right):
    com_x = (shoulder[0] * 0.30 + 
             hip[0] * 0.40 + 
             knee[0] * 0.30)
    
    if facing_right:
        support_center_x = ankle[0] - foot_length * 0.5
    else:
        support_center_x = ankle[0] + foot_length * 0.5
    
    offset_ratio = abs(com_x - support_center_x) / foot_length if foot_length > 0 else 1
    
    # Linear: 0 offset = 100, 2.5 offset = 0
    score = max(0, 100 - offset_ratio * 40)
    return min(100, score)

# Standard 3: Joint Coupling (15%)
def score_joint_coupling(knee_angle, torso_angle):
    predicted_torso = 0.6 * knee_angle + 10
    deviation = abs(torso_angle - predicted_torso)
    # 0° deviation = 100, 60° deviation = 0
    score = max(0, 100 - deviation * (100 / 60))
    return score

# Standard 4: Shoulder Position (15%)
# Shoulder compared to knee position
# Best: shoulder behind knee by <= 0.1 * lower_leg_length
# At knee position: 50 points
# Beyond knee: exponential penalty
def score_shoulder_position(shoulder, knee, lower_leg_length, facing_right):
    if facing_right:
        # shoulder should be LEFT of knee (smaller x)
        # exceed = how far shoulder is to the right of (knee - 0.1L)
        optimal = knee[0] - lower_leg_length * 0.1
        exceed = max(0, shoulder[0] - optimal)
    else:
        # shoulder should be RIGHT of knee (larger x)
        optimal = knee[0] + lower_leg_length * 0.1
        exceed = max(0, optimal - shoulder[0])
    
    exceed_ratio = exceed / lower_leg_length if lower_leg_length > 0 else 0
    
    if exceed_ratio <= 0:
        score = 100
    elif exceed_ratio <= 0.1:
        # Linear from 100 to 50 as exceed goes from 0 to 0.1
        score = 100 - exceed_ratio / 0.1 * 50
    else:
        # Exponential beyond 0.1: at 0.1 = 50, at 0.2 = 33, at 0.3 = 22, at 0.4 = 15
        score = 50 * math.exp(-(exceed_ratio - 0.1) / 0.12)
    
    return min(100, max(0, score))

# Standard 5: Knee Position (15%)
def score_knee_position(knee, ankle, lower_leg_length, facing_right):
    if facing_right:
        exceed = max(0, knee[0] - ankle[0])
    else:
        exceed = max(0, ankle[0] - knee[0])
    
    exceed_ratio = exceed / lower_leg_length if lower_leg_length > 0 else 0
    
    # Tolerance: 0-0.4 = 100 points
    if exceed_ratio <= 0.4:
        score = 100
    else:
        # Beyond 0.4: linear penalty to 0 at 1.2
        penalty = (exceed_ratio - 0.4) / 0.8 * 100
        score = max(0, 100 - penalty)
    
    return min(100, score)

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

    def to_xy(p):
        return (int(p.x * w), int(p.y * h))
    
    left_vis = lm[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_vis = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
    side = "RIGHT" if right_vis > left_vis else "LEFT"
    
    def get(name):
        return lm[getattr(mp_pose.PoseLandmark, f"{side}_{name}").value]
    
    shoulder = to_xy(get("SHOULDER"))
    hip = to_xy(get("HIP"))
    knee = to_xy(get("KNEE"))
    ankle = to_xy(get("ANKLE"))
    
    knee_angle = safe_angle(hip, knee, ankle)
    
    torso_x_offset = abs(shoulder[0] - hip[0])
    torso_vertical = abs(shoulder[1] - hip[1])
    if torso_vertical > 0:
        torso_angle = math.degrees(math.atan2(torso_x_offset, torso_vertical))
    else:
        torso_angle = 0
    
    lower_leg_length = distance(knee, ankle)
    foot_length = lower_leg_length * 0.3 if lower_leg_length > 0 else 1
    
    # Determine facing direction: knee is always in front of hip (toward face)
    facing_right = knee[0] > hip[0]
    
    if facing_right:
        print("FACING DIRECTION: RIGHT")
    else:
        print("FACING DIRECTION: LEFT")
    
    MORANDI_PINK = (180, 150, 200)
    MORANDI_ORANGE = (120, 140, 210)
    MORANDI_GREEN = (110, 160, 130)
    MORANDI_BLUE = (160, 140, 100)
    MORANDI_PURPLE = (180, 120, 160)
    MORANDI_YELLOW = (100, 190, 190)
    
    if knee_angle >= 160:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(image, "FAILED: Knee angle > 160", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, MORANDI_PINK, 2)
        cv2.putText(image, "Not considered as squat", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, MORANDI_GREEN, 2)
        cv2.putText(image, f"Knee angle: {int(knee_angle)}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MORANDI_ORANGE, 2)
        
        print("FAILED: Knee angle > 160° - Not considered as squat")
        
        i = 1
        while os.path.exists(f"output_{i}.jpg"):
            i += 1
        cv2.imwrite(f"output_{i}.jpg", image)
        print(f"Saved output_{i}.jpg")
    
    else:
        score_depth_val = score_depth(knee_angle)
        score_core_val = score_core_balance(shoulder, hip, knee, ankle, foot_length, facing_right)
        score_coupling_val = score_joint_coupling(knee_angle, torso_angle)
        score_shoulder_val = score_shoulder_position(shoulder, knee, lower_leg_length, facing_right)
        score_knee_val = score_knee_position(knee, ankle, lower_leg_length, facing_right)
        
        total = (score_depth_val * 0.40 + 
                 score_core_val * 0.15 + 
                 score_coupling_val * 0.15 + 
                 score_shoulder_val * 0.15 + 
                 score_knee_val * 0.15)
        total = int(total)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(image, f"TOTAL SCORE: {total}/100", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, MORANDI_PINK, 2)
        cv2.putText(image, f"Knee Depth: {int(score_depth_val)}/100", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MORANDI_GREEN, 2)
        cv2.putText(image, f"Core Balance: {int(score_core_val)}/100", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MORANDI_ORANGE, 2)
        cv2.putText(image, f"Joint Coupling: {int(score_coupling_val)}/100", (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MORANDI_BLUE, 2)
        cv2.putText(image, f"Shoulder Pos: {int(score_shoulder_val)}/100", (50, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MORANDI_PURPLE, 2)
        cv2.putText(image, f"Knee Pos: {int(score_knee_val)}/100", (50, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MORANDI_YELLOW, 2)
        
        print(f"Knee angle: {int(knee_angle)}°, Torso angle: {int(torso_angle)}°")
        print(f"Knee Depth: {int(score_depth_val)}/100")
        print(f"Core Balance: {int(score_core_val)}/100")
        print(f"Joint Coupling: {int(score_coupling_val)}/100")
        print(f"Shoulder Position: {int(score_shoulder_val)}/100")
        print(f"Knee Position: {int(score_knee_val)}/100")
        print(f"Total Score: {total}/100")
        
        i = 1
        while os.path.exists(f"output_{i}.jpg"):
            i += 1
        cv2.imwrite(f"output_{i}.jpg", image)
        print(f"Saved output_{i}.jpg")