import cv2
import mediapipe as mp # MediaPipe for human pose estimation (33 body landmarks)
import math
import os

mp_pose = mp.solutions.pose # Pose estimation model
mp_drawing = mp.solutions.drawing_utils # Drawing skeleton landmarks on image

# Math helpers

def calculate_angle(a, b, c):
    ba_x = a[0] - b[0] # Vector ba
    ba_y = a[1] - b[1]
    bc_x = c[0] - b[0] # Vector bc
    bc_y = c[1] - b[1]

    dot = ba_x * bc_x + ba_y * bc_y # Calculate dot product
    mag_ba = math.sqrt(ba_x**2 + ba_y**2) # Calculate length of vector ba and bc
    mag_bc = math.sqrt(bc_x**2 + bc_y**2)

    if mag_ba == 0 or mag_bc == 0: # Prevent division by zero if points overlap
        return 0

    cos_val = dot / (mag_ba * mag_bc) # Cosine of the angle
    cos_val = max(-1, min(1, cos_val)) # Keep cosine value within valid range [-1, 1]
    return math.degrees(math.acos(cos_val)) # Convert to degrees


def calculate_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def get_average_point(landmarks, w, h, left_name, right_name):
    left_id = getattr(mp_pose.PoseLandmark, f"LEFT_{left_name}").value
    right_id = getattr(mp_pose.PoseLandmark, f"RIGHT_{right_name}").value

    left = landmarks[left_id]
    right = landmarks[right_id]

    avg_x = (left.x + right.x) / 2
    avg_y = (left.y + right.y) / 2
    return (int(avg_x * w), int(avg_y * h))

# Feedback system design

def feedback(label, score, good, mid, bad):
    if score >= 80:
        return f"[GOOD] {label}", good
    elif score >= 60:
        return f"[OK] {label}", mid
    else:
        return f"[LOW] {label}", bad


# ============================================================================
# Standard 1: DEPTH
# Purpose: measures squat depth using knee flexion angle
# Key idea: performance is best near an optimal range
# ============================================================================

def score_depth(knee_angle):
    if knee_angle >= 150:
        return 0
    elif knee_angle >= 50: #50 is the most ideal deep-squat knee angle, while both too shallow and too deep will reduce score
        return (150 - knee_angle) / 100 * 100
    else:
        return knee_angle / 50 * 100


# ============================================================================
# Standard 2: CORE
# Purpose: evaluates balance using COM vs foot support
# Key idea: closer COM to center of foot = more stability
# ============================================================================

def score_core(shoulder, hip, knee, ankle, leg_len, facing_right):
    if facing_right:
        cop = ankle[0] + leg_len * 0.35  # Estimated COP offset from ankle based on foot-length to shank-length ratio (~0.65) and COP location within foot (~0.545 of foot length)
    else:
        cop = ankle[0] - leg_len * 0.35

    # Center of mass approximation based on Winter (2009) 8-segment anthropometric model:
    # Head/torso/arms (55%), pelvis (15%), thigh (10% hip + 10% knee), shank (4.5% knee + 4.5% ankle), foot (1% at ankle)

    com = (shoulder[0]*0.55 + hip[0]*0.25 +
           knee[0]*0.145 + ankle[0]*0.055)

    foot_len = leg_len * 0.65  # A human being's average foot length is set to be 0.65 * leg length
    offset = abs(com - cop) / foot_len if foot_len > 0 else 1  # The difference of Estimated COP and COM will reduce score

    return max(0, 100 - offset * 40)


# ============================================================================
# Standard 3: COORDINATION
# Purpose: checks timing between torso lean and knee bend
# Key idea: good squat = torso and knee move in synchronized
# ============================================================================

def score_coordination(knee_angle, torso_angle):
    
    # SCIENTIFIC BASIS: Maddox, Sievert & Bennett 2020, Journal of Biomechanics
    # Coupling is non-linear: 
    # Deep squat (knee<70°) trunk lean 30-40°
    # Parallel squat (knee~90°) trunk lean 15-25°
    # Shallow squat (knee>110°) trunk lean 5-15°
    
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
# Standard 4: POSTURE
# Purpose: evaluates forward/backward torso position
# Key idea: shoulders should stay controlled relative to knee
# ============================================================================

def score_posture(shoulder, knee, leg_len, facing_right):

    #ideal shoulder position is set to be posterior or equals to (0.1 * leg length behind the knee), if anterior to this borderline will reduce score 
    
    ideal = leg_len * 0.1

    if facing_right:
        diff = knee[0] - shoulder[0] 
    else:
        diff = shoulder[0] - knee[0]

    if diff >= ideal:
        return 100
    else:
        return max(0, 70 + (diff / ideal) * 30)


# ============================================================================
# Standard 5: HIP LOAD
# Purpose: estimates hip vs knee dominance
# Key idea: hip-dominant squat = safer joint loading
# ============================================================================

def score_hip_load(hip, knee, ankle, facing_right):
    gravity = hip[0]*0.4 + knee[0]*0.35 + ankle[0]*0.25

    hip_lever = abs(gravity - hip[0])
    knee_lever = abs(gravity - knee[0])

    if knee_lever == 0:
        return 50

    # SCIENTIFIC BASIS: Di Paolo et al. 2024, The Knee
    # Hip/Knee extensor moment ratio is key indicator of squat quality
    # Higher ratio = better hip-dominant technique

    ratio = hip_lever / knee_lever  

    if ratio >= 1.0:
        return 100
    elif ratio >= 0.7:
        return 70 + (ratio - 0.7) / 0.3 * 30
    elif ratio >= 0.5:
        return 40 + (ratio - 0.5) / 0.2 * 30
    else:
        return ratio / 0.5 * 40


# ============================================================================
# Standard 6: KNEE TRACK
# Purpose: checks knee alignment over foot
# Key idea: knee should not collapse too far past toes
# ============================================================================

def score_knee_track(knee, ankle, leg_len, facing_right, knee_angle):
    foot_len = leg_len * 0.65

    if facing_right:
        over = max(0, knee[0] - ankle[0])
    else:
        over = max(0, ankle[0] - knee[0])

    ratio = over / foot_len if foot_len > 0 else 0

    if knee_angle < 90:
        ratio = ratio / 1.2
    elif knee_angle > 120:
        ratio = ratio / 0.8

    # SCIENTIFIC BASIS: Fry et al. 2003, Swinton et al. 2012
    # Restricting knee over toe reduces knee stress 22-28%, BUT increases hip stress ~1000%
    # He observed that tolerance up to 0.3 foot length is normal
    # I set the tolerance up to 0.4 considering it's a deep squat which needs more hip involvement and the score range is more distinguishable on sample pictures

    if ratio <= 0.4:
        return 100
    elif ratio <= 0.8:
        return 100 - (ratio - 0.4) / 0.4 * 40  # Linear penalty between 0.4-0.8
    else:
        return 60 * math.exp(-(ratio - 0.8) / 0.3)  # Exponential decay beyond 0.8 because it's too dangerous


# ============================================================================
# MAIN
# ============================================================================

filename = input("Enter image filename: ")
img = cv2.imread(filename)

if img is None:
    print("Image not found")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert to RGB (MediaPipe requirement)

with mp_pose.Pose(static_image_mode=True) as pose: # static_image_mode=True: process a single image, not a video
    results = pose.process(img_rgb) # Returns a PoseLandmark object containing all detected points

    if not results.pose_landmarks:
        print("No person detected")
        exit()

    h, w, _ = img.shape
    lm = results.pose_landmarks.landmark  # Extract 33 body landmarks

    shoulder = get_average_point(lm, w, h, "SHOULDER", "SHOULDER")  # Get the average of left and right
    hip = get_average_point(lm, w, h, "HIP", "HIP")
    knee = get_average_point(lm, w, h, "KNEE", "KNEE")
    ankle = get_average_point(lm, w, h, "ANKLE", "ANKLE")

    knee_angle = calculate_angle(hip, knee, ankle) # Knee flexion angle computed from hip–knee–ankle joint chain
    leg_len = calculate_distance(knee, ankle)

    dx = abs(shoulder[0] - hip[0])  # horizontal displacement
    dy = abs(shoulder[1] - hip[1])  # vertical displacement
    torso_angle = math.degrees(math.atan2(dx, dy)) if dy > 0 else 0  # Convert vector ratio into angle using arctangent

    # No matter it's a standard or not standard squat, the knee will always be closer to the face than the hip
    # So determine body orientation (left/right facing) based on the knee-hip ordering
    
    facing_right = knee[0] > hip[0] 

    if knee_angle >= 150: # Knee_angle >= 150 will result in 0 because it's not considered as a squat
        print("Not a squat")
        exit()

    # Scores
    s1 = score_depth(knee_angle)
    s2 = score_core(shoulder, hip, knee, ankle, leg_len, facing_right)
    s3 = score_coordination(knee_angle, torso_angle)
    s4 = score_posture(shoulder, knee, leg_len, facing_right)
    s5 = score_hip_load(hip, knee, ankle, facing_right)
    s6 = score_knee_track(knee, ankle, leg_len, facing_right, knee_angle)

    total = int(s1*0.25 + s2*0.15 + s3*0.20 + s4*0.15 + s5*0.10 + s6*0.15)

    # Print score

    print(f"\nFacing: {'RIGHT' if facing_right else 'LEFT'}")
    print(f"Knee: {int(knee_angle)} Torso: {int(torso_angle)}")

    print("\nSCORES:")
    print(f"Depth: {int(s1)}")
    print(f"Core: {int(s2)}")
    print(f"Coordination: {int(s3)}")
    print(f"Posture: {int(s4)}")
    print(f"Hip Load: {int(s5)}")
    print(f"Knee Track: {int(s6)}")

    print(f"\nTOTAL: {total}/100")

    # Print feedback

    print("\nFEEDBACK:")

    tests = [
        ("Depth", s1, "Good depth control", "Slightly shallow", "Too shallow"),
        ("Core", s2, "Stable core", "Moderate instability", "Poor balance"),
        ("Coordination", s3, "Good coordination", "Timing off", "Poor coordination"),
        ("Posture", s4, "Good posture", "Slight lean", "Excessive lean"),
        ("Hip Load", s5, "Hip dominant", "Moderate knee use", "Knee dominant"),
        ("Knee Track", s6, "Safe knee position", "Slight forward knee", "Unsafe knee position")
    ]

    # Iterate through all squat evaluation standards
    # Each entry contains:
    # (metric name, score, good feedback, medium feedback, bad feedback)

    for name, score, g, m, b in tests:
    
        # Convert numeric score into categorical feedback using rule-based thresholds
        status, msg = feedback(name, score, g, m, b)
    
        # Print result label (e.g., [GOOD], [OK], [LOW])
        print(status)
    
        # Only print explanation message if it exists (avoid empty output)
        if msg:
            print(" ->", msg)

    # Image text output

    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.putText(img, f"TOTAL: {total}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (180,150,200), 2)

    cv2.putText(img, f"Depth: {int(s1)}", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110,160,130), 2)

    cv2.putText(img, f"Core: {int(s2)}", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,140,100), 2)

    cv2.putText(img, f"Coord: {int(s3)}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,190,190), 2)

    cv2.putText(img, f"Posture: {int(s4)}", (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,140,210), 2)

    cv2.putText(img, f"HipLoad: {int(s5)}", (50, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,120,160), 2)

    cv2.putText(img, f"KneeTrk: {int(s6)}", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,200), 2)

    # Save the image as a new file
    i = 1
    while os.path.exists(f"output_{i}.jpg"):
        i += 1

    cv2.imwrite(f"output_{i}.jpg", img)
    print(f"\nSaved output_{i}.jpg")