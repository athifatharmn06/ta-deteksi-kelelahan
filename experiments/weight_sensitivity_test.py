import numpy as np

def calculate_score(ear, mar, head, weights):
    # ear, mar, head are normalized features (0-1 where 1 is drowsy)
    return weights[0]*ear + weights[1]*mar + weights[2]*head

# Case 1: Driver Talking (High MAR excursion, but no eye/head drift)
# Normalized: EAR=0.2 (alert), MAR=0.7 (talking), Head=0.1 (center)
talking_ear = 0.2
talking_mar = 0.7
talking_head = 0.1

# Case 2: Driver Drowsy (Micro-blink, Moderate MAR)
# Normalized: EAR=0.6 (drooping), MAR=0.4 (gaping), Head=0.5 (slumping)
drowsy_ear = 0.6
drowsy_mar = 0.4
drowsy_head = 0.5

# Weight sets
proposed_weights = [0.45, 0.25, 0.30] # eye, mouth, head
equal_weights = [0.333, 0.333, 0.333]

# Results
score_talking_proposed = calculate_score(talking_ear, talking_mar, talking_head, proposed_weights)
score_talking_equal = calculate_score(talking_ear, talking_mar, talking_head, equal_weights)

score_drowsy_proposed = calculate_score(drowsy_ear, drowsy_mar, drowsy_head, proposed_weights)
score_drowsy_equal = calculate_score(drowsy_ear, drowsy_mar, drowsy_head, equal_weights)

# Threshold for alert = 0.45 (Hypothetical)
thresh = 0.45

print(f"Alert Threshold: {thresh}")
print(f"Case 1: Talking (EAR=0.2, MAR=0.7, Head=0.1)")
print(f"  Proposed Score: {score_talking_proposed:.3f} -> {'ALERT' if score_talking_proposed > thresh else 'SAFE'}")
print(f"  Equal Score:    {score_talking_equal:.3f} -> {'ALERT' if score_talking_equal > thresh else 'SAFE'}")

print(f"\nCase 2: Drowsy (EAR=0.6, MAR=0.4, Head=0.5)")
print(f"  Proposed Score: {score_drowsy_proposed:.3f} -> {'ALERT' if score_drowsy_proposed > thresh else 'SAFE'}")
print(f"  Equal Score:    {score_drowsy_equal:.3f} -> {'ALERT' if score_drowsy_equal > thresh else 'SAFE'}")
