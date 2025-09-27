################################################ Import Libraries  ##########################################
import cv2
import sys
import os
import matplotlib
import numpy as np
from collections import Counter
import face_recognition
import dlib
from math import hypot
import time

from object_detection import yoloV3Detect
from landmark_models import *
from face_spoofing import *
from headpose_estimation import *
from face_detection import get_face_detector, find_faces

# Import the audio detection module
from audio_detection import AudioDetector

################################################ Setup  ######################################################

# face recognition
l = os.listdir('student_db')
known_face_encodings = []
known_face_names = []
face_locations = []
face_encodings = []
face_names = []

for image in l:
    obama_image = face_recognition.load_image_file('student_db/'+image)
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    known_face_encodings.append(obama_face_encoding)
    known_face_names.append(image.split('.')[0])

# headpose model
h_model = load_hp_model('models/Headpose_customARC_ZoomShiftNoise.hdf5')

# face detection model
face_model = get_face_detector()

# face landmark model
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Audio Detection Setup
print("Initializing audio detection system...")
audio_detector = AudioDetector(
    threshold_db=15,  # Adjust based on your environment
    speech_threshold=0.1,
    suspicious_words=[
        'answer', 'help', 'question', 'tell me', 'what is',
        'google', 'search', 'phone', 'call', 'message',
        'copy', 'paste', 'share', 'send', 'receive',
        'solution', 'hint', 'explain', 'show me'
    ]
)

# Start audio monitoring
if audio_detector.start_monitoring():
    print("Audio monitoring started successfully!")
else:
    print("Warning: Audio monitoring failed to start")

# Others
video_capture = cv2.VideoCapture(0)
process_this_frame = False
no_of_frames_0 = 0
no_of_frames_1 = 0
no_of_frames_2 = 0
no_of_frames_3 = 0
no_of_frames_4 = 0
no_of_frames_5 = 0
no_of_frames_6 = 0
no_of_frames_7 = 0
no_of_frames_8 = 0  # Audio detection frames
no_of_frames_9 = 0  # Suspicious speech frames
font = cv2.FONT_HERSHEY_PLAIN
flag = True

# Alert tracking variables
alert_count = 0
max_alerts = 5
result = "result:[SAFE]"  # Default status

#################################################### ALERT #####################################################
def alert(condition, no_of_frames):
    if(condition):
        no_of_frames = no_of_frames + 1
    else:
        no_of_frames = 0
    return no_of_frames

def check_and_report_alert(alert_type, frames_count, threshold, global_alert_count):
    """Check if alert condition is met and report to terminal"""
    if frames_count == threshold + 1:  # Only trigger on the exact frame when threshold is crossed
        global_alert_count += 1
        return True, global_alert_count
    return False, global_alert_count

#################################################### MAIN #####################################################

print("Starting proctoring system...")
print("Press Ctrl+C to quit")
print("System will automatically stop after 5 alerts")

try:
    while True:
        # frame skipping to save time
        process_this_frame = not process_this_frame 

        # Grab a single frame of video
        ret, frame = video_capture.read()

        frame2 = frame.copy()
        frame3 = frame.copy()
      
        # Audio processing should happen every frame for faster detection
        # Get audio status
        audio_status = audio_detector.get_audio_status()

        # Reset result to SAFE at the beginning of each frame
        result = "result:[SAFE]"

        ##### Audio Detection (Process every frame for faster response) #####
        
        # Audio Level Detection
        audio_condition = (audio_status['db_level'] > 20)  # High audio level
        no_of_frames_8 = alert(audio_condition, no_of_frames_8)

        # Alert Check - Changed from 3 to 1 for faster detection
        alert_triggered, alert_count = check_and_report_alert("High Audio Level", no_of_frames_8, 1, alert_count)
        if alert_triggered:
            result = f"result:[ALERT-High Audio Level] ({alert_count})"

        # Suspicious Speech Detection
        suspicious_condition = audio_status['suspicious_speech']
        no_of_frames_9 = alert(suspicious_condition, no_of_frames_9)

        # Alert Check - Changed from 5 to 1 for faster detection
        alert_triggered, alert_count = check_and_report_alert("Suspicious Speech", no_of_frames_9, 1, alert_count)
        if alert_triggered:
            result = f"result:[ALERT-Suspicious Speech] ({alert_count})"

        # Functionalities
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            try:
                ##### Object Detection #####
                try:
                    fboxes, fclasses = yoloV3Detect(small_frame)
                    
                    to_detect = ['person', 'laptop', 'cell phone', 'book', 'tv']
                    temp1, temp2 = [], []

                    for i in range(len(fclasses)):
                        if(fclasses[i] in to_detect):
                            temp1.append(fboxes[i])
                            temp2.append(fclasses[i])
                 
                    # Counter
                    count_items = Counter(temp2)
                except Exception as e:
                    count_items = {}
                    count_items['person'] = 0
                    count_items['laptop'] = 0
                    count_items['cell phone'] = 0
                    count_items['book'] = 0
                    count_items['tv'] = 0
                    print(e)

                # Multiple Person Buffer
                condition = (count_items['person'] != 1)
                no_of_frames_0 = alert(condition, no_of_frames_0)

                # Alert Check
                alert_triggered, alert_count = check_and_report_alert("Multiple People Detected", no_of_frames_0, 10, alert_count)
                if alert_triggered:
                    result = f"result:[ALERT-Multiple People Detected] ({alert_count})"
                
                # Object Detection Buffer
                condition = (count_items['laptop'] >= 1 or 
                            count_items['cell phone'] >= 1 or 
                            count_items['book'] >= 1 or 
                            count_items['tv'] >= 1)
             
                no_of_frames_1 = alert(condition, no_of_frames_1)

                # Alert Check
                alert_triggered, alert_count = check_and_report_alert("Banned Objects Detected", no_of_frames_1, 2, alert_count)
                if alert_triggered:
                    result = f"result:[ALERT-Banned Objects Detected] ({alert_count})"

                ##### Audio Detection #####
                
                # Audio Level Detection
                audio_condition = (audio_status['db_level'] > 20)  # High audio level
                no_of_frames_8 = alert(audio_condition, no_of_frames_8)

                # Alert Check - Changed from 3 to 1 for faster detection
                alert_triggered, alert_count = check_and_report_alert("High Audio Level", no_of_frames_8, 1, alert_count)
                if alert_triggered:
                    result = f"result:[ALERT-High Audio Level] ({alert_count})"

                # Suspicious Speech Detection
                suspicious_condition = audio_status['suspicious_speech']
                no_of_frames_9 = alert(suspicious_condition, no_of_frames_9)

                # Alert Check - Changed from 5 to 1 for faster detection
                alert_triggered, alert_count = check_and_report_alert("Suspicious Speech", no_of_frames_9, 1, alert_count)
                if alert_triggered:
                    result = f"result:[ALERT-Suspicious Speech] ({alert_count})"

                if(count_items['person'] == 1):

                    #### face detection using caffe model of OpenCV's DNN module ####
                    
                    # detect face
                    faces = find_faces(small_frame, face_model)
                    if len(faces) > 0:
                        face = faces[0]
                    else:
                        condition = (len(faces) < 1)
                        no_of_frames_7 = alert(condition, no_of_frames_7)

                        # Alert Check
                        alert_triggered, alert_count = check_and_report_alert("No Face Detected", no_of_frames_7, 10, alert_count)
                        if alert_triggered:
                            result = f"result:[ALERT-No Face Detected] ({alert_count})"
                        
                        # Print result to terminal and continue to next frame
                        print(result, flush=True)
                        
                        # Check if max alerts reached
                        if alert_count >= max_alerts:
                            print(f"\n\nMAXIMUM ALERTS REACHED ({max_alerts})! STOPPING PROCTORING SYSTEM!")
                            break
                        
                        continue
                   
                    if(flag == True):
                        #### face verification using face_recognition library ####
                        
                        # modifying order
                        face_locations = [[face[1], face[2], face[3], face[0]]]
                       
                        # Convert BGR image to RGB image (which uses)
                        rgb_small_frame = small_frame[:, :, ::-1]

                        # get CNN feature vector
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                        # get similarity
                        face_encoding = face_encodings[0]
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                        else:
                            name = "Unknown"
                        flag = False
                    
                    # Buffer
                    condition = (name == 'Unknown')  
                    no_of_frames_2 = alert(condition, no_of_frames_2)

                    # Alert Check
                    alert_triggered, alert_count = check_and_report_alert("Unknown Face", no_of_frames_2, 10, alert_count)
                    if alert_triggered:
                        result = f"result:[ALERT-Unknown Face] ({alert_count})"
                       
                    #### mouth movement ####

                    # get landmarks
                    left = face[0]*4
                    top = face[1]*4
                    right = face[2]*4
                    bottom = face[3]*4
                    face_dlib = dlib.rectangle(left, top, right, bottom)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    facial_landmarks = predictor(gray, face_dlib)

                    mouth_ratio = get_mouth_ratio([60, 62, 64, 66], frame2, facial_landmarks)
                    
                    # Buffer
                    condition = (mouth_ratio > 0.1)
                    no_of_frames_4 = alert(condition, no_of_frames_4)

                    # Alert Check
                    alert_triggered, alert_count = check_and_report_alert("Mouth Open", no_of_frames_4, 10, alert_count)
                    if alert_triggered:
                        result = f"result:[ALERT-Mouth Open] ({alert_count})"

                    #### head pose ####
                    oAnglesNp, oBboxExpanded = headpose_inference(h_model, frame2, face)
                    
                    # Buffer
                    condition1 = (round(oAnglesNp[0], 1) not in [0.0, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7] and 
                                round(oAnglesNp[1], 0) not in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                    no_of_frames_5 = alert(condition1, no_of_frames_5)

                    # Alert Check
                    alert_triggered, alert_count = check_and_report_alert("Looking Away - Head Pose", no_of_frames_5, 10, alert_count)
                    if alert_triggered:
                        result = f"result:[ALERT-Looking Away - Head Pose] ({alert_count})"
                
                    ##### Blinking (to support down eye tracking) ######

                    left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], frame2, facial_landmarks)
                    right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], frame2, facial_landmarks)
                    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                    
                    ##### eye tracker #####

                    gaze_ratio1_left_eye, gaze_ratio2_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], frame2, facial_landmarks)
                    gaze_ratio1_right_eye, gaze_ratio2_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], frame2, facial_landmarks)

                    # Left/Right
                    gaze_ratio1 = (gaze_ratio1_right_eye + gaze_ratio1_left_eye) / 2

                    # Buffer
                    condition = (gaze_ratio1 <= 0.35 or gaze_ratio1 >= 4 or condition1 == True)
                    no_of_frames_3 = alert(condition, no_of_frames_3)

                    # Alert Check
                    alert_triggered, alert_count = check_and_report_alert("Looking Away - Eye Tracking", no_of_frames_3, 10, alert_count)
                    if alert_triggered:
                        result = f"result:[ALERT-Looking Away - Eye Tracking] ({alert_count})"
                        
                    #### face spoofing ####
                    measures = face_spoof(frame2, face)

                    # Buffer
                    condition = (np.mean(measures) < 0.7)
                    no_of_frames_6 = alert(condition, no_of_frames_6)
                    
                    # Alert Check
                    alert_triggered, alert_count = check_and_report_alert("Spoof Face", no_of_frames_6, 10, alert_count)
                    if alert_triggered:
                        result = f"result:[ALERT-Spoof Face] ({alert_count})"
                
                else:
                    flag = True

                # Print result to terminal with newline to avoid buffer issues
                print(result, flush=True)
                
                # Check if max alerts reached
                if alert_count >= max_alerts:
                    print(f"\n\nMAXIMUM ALERTS REACHED ({max_alerts})! STOPPING PROCTORING SYSTEM!")
                    break

            except Exception as e:
                print(e) 
                flag = True

        # Small delay to prevent excessive CPU usage
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Cleanup
    print("\nCleaning up resources...")
    
    # Stop audio monitoring
    audio_detector.cleanup()
    
    # Release handle to the webcam
    video_capture.release()
    
    print("Cleanup completed.")
    
    if alert_count >= max_alerts:
        print(f"FINAL STATUS: SYSTEM STOPPED DUE TO {alert_count} ALERTS")
    else:
        print("Proctoring system terminated successfully!")