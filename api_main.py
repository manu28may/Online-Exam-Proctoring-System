from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import sqlite3
from datetime import datetime
import os
import logging
from collections import Counter
import face_recognition
import dlib
from math import hypot
import time

# Import your existing detection modules
from object_detection import yoloV3Detect
from landmark_models import get_mouth_ratio, get_blinking_ratio, get_gaze_ratio
from face_spoofing import face_spoof
from headpose_estimation import headpose_inference, load_hp_model
from face_detection import get_face_detector, find_faces
from audio_detection import AudioDetector

app = FastAPI(title="Cheating Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models (same as your main file)
print("Loading models...")

# Face recognition setup
known_face_encodings = []
known_face_names = []

try:
    l = os.listdir('student_db')
    for image in l:
        if image.endswith(('.jpg', '.jpeg', '.png')):
            obama_image = face_recognition.load_image_file('student_db/'+image)
            if face_recognition.face_encodings(obama_image):
                obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
                known_face_encodings.append(obama_face_encoding)
                known_face_names.append(image.split('.')[0])
    print(f"Loaded {len(known_face_names)} student faces")
except Exception as e:
    print(f"Error loading student faces: {e}")

# Load models
try:
    h_model = load_hp_model('models/Headpose_customARC_ZoomShiftNoise.hdf5')
    face_model = get_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

class ProctoringSessions:
    def __init__(self):
        self.active_sessions = {}
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing results"""
        conn = sqlite3.connect('proctoring_results.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                student_name TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_alerts INTEGER,
                status TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP,
                result_type TEXT,
                result_value TEXT,
                alert_count INTEGER,
                FOREIGN KEY (session_id) REFERENCES monitoring_sessions (session_id)
            )
        ''')
        conn.commit()
        conn.close()

proctoring_sessions = ProctoringSessions()

class DetectionProcessor:
    def __init__(self):
        self.alert_count = 0
        self.max_alerts = 5
        self.session_active = True
        self.flag = True
        
        # Frame counters (same as your main file)
        self.no_of_frames_0 = 0  # Multiple people
        self.no_of_frames_1 = 0  # Banned objects
        self.no_of_frames_2 = 0  # Unknown face
        self.no_of_frames_3 = 0  # Looking away - eye tracking
        self.no_of_frames_4 = 0  # Mouth open
        self.no_of_frames_5 = 0  # Looking away - head pose
        self.no_of_frames_6 = 0  # Spoof face
        self.no_of_frames_7 = 0  # No face detected
        self.no_of_frames_8 = 0  # Audio detection
        self.no_of_frames_9 = 0  # Suspicious speech
        
        # Initialize audio detector
        self.audio_detector = AudioDetector(
            threshold_db=15,
            speech_threshold=0.1,
            suspicious_words=[
                'answer', 'help', 'question', 'tell me', 'what is',
                'google', 'search', 'phone', 'call', 'message'
            ]
        )
        self.audio_detector.start_monitoring()
    
    def alert(self, condition, no_of_frames):
        """Alert function from your main file"""
        if condition:
            no_of_frames = no_of_frames + 1
        else:
            no_of_frames = 0
        return no_of_frames

    def check_and_report_alert(self, alert_type, frames_count, threshold):
        """Check if alert condition is met"""
        if frames_count == threshold + 1:
            self.alert_count += 1
            return True
        return False
        
    def process_frame(self, frame_data: str, session_id: str):
        """Process frame using your existing detection logic"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {'error': 'Invalid frame data'}
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'detections': []
            }
            
            frame2 = frame.copy()
            frame3 = frame.copy()
            result = "result:[SAFE]"
            
            # Resize frame (same as your main file)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Get audio status
            audio_status = self.audio_detector.get_audio_status()
            
            ##### Object Detection (Your existing logic) #####
            try:
                fboxes, fclasses = yoloV3Detect(small_frame)
                
                to_detect = ['person', 'laptop', 'cell phone', 'book', 'tv']
                temp1, temp2 = [], []

                for i in range(len(fclasses)):
                    if fclasses[i] in to_detect:
                        temp1.append(fboxes[i])
                        temp2.append(fclasses[i])
             
                count_items = Counter(temp2)
                
                # Set default counts
                for item in to_detect:
                    if item not in count_items:
                        count_items[item] = 0
                        
            except Exception as e:
                count_items = Counter({'person': 0, 'laptop': 0, 'cell phone': 0, 'book': 0, 'tv': 0})
                print(f"Object detection error: {e}")

            # Store object detection results
            detected_objects = [item for item, count in count_items.items() if count > 0 and item != 'person']
            results['detections'].append({
                'type': 'object_detection',
                'result': detected_objects
            })
            print(f"result2:{detected_objects}")

            ##### Multiple Person Detection #####
            condition = (count_items['person'] != 1)
            self.no_of_frames_0 = self.alert(condition, self.no_of_frames_0)
            
            if self.check_and_report_alert("Multiple People Detected", self.no_of_frames_0, 10):
                result = f"result:[ALERT-Multiple People Detected] ({self.alert_count})"

            ##### Banned Objects Detection #####
            condition = (count_items['laptop'] >= 1 or 
                        count_items['cell phone'] >= 1 or 
                        count_items['book'] >= 1 or 
                        count_items['tv'] >= 1)
         
            self.no_of_frames_1 = self.alert(condition, self.no_of_frames_1)

            if self.check_and_report_alert("Banned Objects Detected", self.no_of_frames_1, 2):
                result = f"result:[ALERT-Banned Objects Detected] ({self.alert_count})"

            ##### Audio Detection #####
            audio_condition = (audio_status['db_level'] > 20)
            self.no_of_frames_8 = self.alert(audio_condition, self.no_of_frames_8)

            if self.check_and_report_alert("High Audio Level", self.no_of_frames_8, 1):
                result = f"result:[ALERT-High Audio Level] ({self.alert_count})"

            suspicious_condition = audio_status['suspicious_speech']
            self.no_of_frames_9 = self.alert(suspicious_condition, self.no_of_frames_9)

            if self.check_and_report_alert("Suspicious Speech", self.no_of_frames_9, 1):
                result = f"result:[ALERT-Suspicious Speech] ({self.alert_count})"

            ##### Face Processing (only if one person detected) #####
            if count_items['person'] == 1:
                try:
                    # Face detection
                    faces = find_faces(small_frame, face_model)
                    
                    if len(faces) == 0:
                        condition = True
                        self.no_of_frames_7 = self.alert(condition, self.no_of_frames_7)
                        
                        if self.check_and_report_alert("No Face Detected", self.no_of_frames_7, 10):
                            result = f"result:[ALERT-No Face Detected] ({self.alert_count})"
                    else:
                        face = faces[0]
                        
                        ##### Face Recognition #####
                        if self.flag and len(known_face_encodings) > 0:
                            face_locations = [[face[1], face[2], face[3], face[0]]]
                            rgb_small_frame = small_frame[:, :, ::-1]
                            
                            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                            
                            if face_encodings:
                                face_encoding = face_encodings[0]
                                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                
                                if matches[best_match_index]:
                                    name = known_face_names[best_match_index]
                                else:
                                    name = "Unknown"
                                self.flag = False
                            else:
                                name = "Unknown"
                        elif len(known_face_encodings) == 0:
                            name = "Unknown"  # No registered students
                        
                        # Unknown face buffer
                        condition = (name == 'Unknown')
                        self.no_of_frames_2 = self.alert(condition, self.no_of_frames_2)
                        
                        if self.check_and_report_alert("Unknown Face", self.no_of_frames_2, 10):
                            result = f"result:[ALERT-Unknown Face] ({self.alert_count})"

                        ##### Facial Analysis (if face detected) #####
                        if len(faces) > 0:
                            try:
                                # Mouth movement
                                left = face[0]*4
                                top = face[1]*4
                                right = face[2]*4
                                bottom = face[3]*4
                                face_dlib = dlib.rectangle(left, top, right, bottom)
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                facial_landmarks = predictor(gray, face_dlib)

                                mouth_ratio = get_mouth_ratio([60, 62, 64, 66], frame2, facial_landmarks)
                                
                                condition = (mouth_ratio > 0.1)
                                self.no_of_frames_4 = self.alert(condition, self.no_of_frames_4)

                                if self.check_and_report_alert("Mouth Open", self.no_of_frames_4, 10):
                                    result = f"result:[ALERT-Mouth Open] ({self.alert_count})"

                                # Head pose
                                oAnglesNp, oBboxExpanded = headpose_inference(h_model, frame2, face)
                                
                                condition1 = (round(oAnglesNp[0], 1) not in [0.0, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7] and 
                                            round(oAnglesNp[1], 0) not in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                                self.no_of_frames_5 = self.alert(condition1, self.no_of_frames_5)

                                if self.check_and_report_alert("Looking Away - Head Pose", self.no_of_frames_5, 10):
                                    result = f"result:[ALERT-Looking Away - Head Pose] ({self.alert_count})"

                                # Eye tracking
                                left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], frame2, facial_landmarks)
                                right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], frame2, facial_landmarks)
                                
                                gaze_ratio1_left_eye, gaze_ratio2_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], frame2, facial_landmarks)
                                gaze_ratio1_right_eye, gaze_ratio2_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], frame2, facial_landmarks)
                                gaze_ratio1 = (gaze_ratio1_right_eye + gaze_ratio1_left_eye) / 2

                                condition = (gaze_ratio1 <= 0.35 or gaze_ratio1 >= 4 or condition1 == True)
                                self.no_of_frames_3 = self.alert(condition, self.no_of_frames_3)

                                if self.check_and_report_alert("Looking Away - Eye Tracking", self.no_of_frames_3, 10):
                                    result = f"result:[ALERT-Looking Away - Eye Tracking] ({self.alert_count})"

                                # Face spoofing
                                measures = face_spoof(frame2, face)
                                condition = (np.mean(measures) < 0.7)
                                self.no_of_frames_6 = self.alert(condition, self.no_of_frames_6)
                                
                                if self.check_and_report_alert("Spoof Face", self.no_of_frames_6, 10):
                                    result = f"result:[ALERT-Spoof Face] ({self.alert_count})"
                                    
                            except Exception as e:
                                print(f"Facial analysis error: {e}")
                                
                except Exception as e:
                    print(f"Face processing error: {e}")
                    self.flag = True
            else:
                self.flag = True

            # Store detection result
            results['detections'].append({
                'type': 'main_result',
                'result': result,
                'alert_count': self.alert_count
            })
            
            # Print to console (same format as your main file)
            print(result, flush=True)
            
            # Check if maximum alerts reached
            if self.alert_count >= self.max_alerts:
                self.session_active = False
                results['session_status'] = 'TERMINATED'
                results['message'] = f'MAXIMUM ALERTS REACHED ({self.max_alerts})! STOPPING PROCTORING SYSTEM!'
                print(f"MAXIMUM ALERTS REACHED ({self.max_alerts})! STOPPING PROCTORING SYSTEM!")
            
            # Store result in database
            self.store_result(session_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {'error': str(e)}
    
    def store_result(self, session_id: str, results: dict):
        """Store detection results in database"""
        try:
            conn = sqlite3.connect('proctoring_results.db')
            cursor = conn.cursor()
            
            for detection in results['detections']:
                cursor.execute('''
                    INSERT INTO detection_results 
                    (session_id, timestamp, result_type, result_value, alert_count)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    results['timestamp'],
                    detection['type'],
                    detection['result'] if isinstance(detection['result'], str) else str(detection['result']),
                    detection.get('alert_count', 0)
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """API Testing Dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔍 Cheating Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
            .endpoints { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .endpoint { margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #007bff; }
            button { padding: 12px 24px; margin: 5px; font-size: 16px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #218838; }
            .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
            .safe { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .alert { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Cheating Detection API</h1>
                <p>Integration with your existing proctoring system</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>API Status</h3>
                    <p>✅ Running on Port 8000</p>
                </div>
                <div class="stat-card">
                    <h3>Models Loaded</h3>
                    <p>✅ Face, Object, Audio, Headpose</p>
                </div>
                <div class="stat-card">
                    <h3>Database</h3>
                    <p>✅ SQLite Connected</p>
                </div>
            </div>
            
            <div class="endpoints">
                <h3>📋 Available API Endpoints:</h3>
                <div class="endpoint"><strong>GET /api/health</strong> - Check API health status</div>
                <div class="endpoint"><strong>POST /api/start-session</strong> - Start new monitoring session</div>
                <div class="endpoint"><strong>GET /api/sessions</strong> - Get all sessions list</div>
                <div class="endpoint"><strong>GET /api/session/{id}/results</strong> - Get specific session results</div>
                <div class="endpoint"><strong>POST /api/upload-student-photo</strong> - Upload student photo to database</div>
                <div class="endpoint"><strong>POST /api/test-detection</strong> - Test detection without camera</div>
                <div class="endpoint"><strong>WebSocket /ws/monitor/{session_id}</strong> - Real-time monitoring</div>
            </div>
            
            <button onclick="testHealth()">🔍 Test API Health</button>
            <button onclick="testDetection()">🎯 Test Detection</button>
            <button onclick="viewDocs()">📚 View API Docs</button>
            
            <div id="result"></div>
            
            <script>
                async function testHealth() {
                    try {
                        const response = await fetch('/api/health');
                        const data = await response.json();
                        document.getElementById('result').innerHTML = 
                            '<div class="result safe"><strong>API Health:</strong><br>' + JSON.stringify(data, null, 2) + '</div>';
                    } catch (error) {
                        document.getElementById('result').innerHTML = 
                            '<div class="result alert"><strong>Error:</strong> ' + error.message + '</div>';
                    }
                }
                
                async function testDetection() {
                    try {
                        const response = await fetch('/api/test-detection?student_name=TestStudent', {method: 'POST'});
                        const data = await response.json();
                        document.getElementById('result').innerHTML = 
                            '<div class="result safe"><strong>Detection Test:</strong><br>' + JSON.stringify(data, null, 2) + '</div>';
                    } catch (error) {
                        document.getElementById('result').innerHTML = 
                            '<div class="result alert"><strong>Error:</strong> ' + error.message + '</div>';
                    }
                }
                
                function viewDocs() {
                    window.open('/docs', '_blank');
                }
            </script>
        </div>
    </body>
    </html>
    """
    return html_content

@app.post("/api/start-session")
async def start_session(student_name: str):
    """Start a new proctoring session"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    conn = sqlite3.connect('proctoring_results.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO monitoring_sessions (session_id, student_name, start_time, status)
        VALUES (?, ?, ?, ?)
    ''', (session_id, student_name, datetime.now(), 'ACTIVE'))
    conn.commit()
    conn.close()
    
    proctoring_sessions.active_sessions[session_id] = {
        'student_name': student_name,
        'start_time': datetime.now(),
        'alert_count': 0,
        'status': 'ACTIVE'
    }
    
    return {
        'session_id': session_id,
        'message': f'Session started for {student_name}',
        'status': 'SUCCESS'
    }

@app.get("/api/sessions")
async def get_all_sessions():
    """Get all monitoring sessions"""
    conn = sqlite3.connect('proctoring_results.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM monitoring_sessions ORDER BY start_time DESC')
    sessions = cursor.fetchall()
    conn.close()
    
    session_list = []
    for session in sessions:
        session_list.append({
            'id': session[0],
            'session_id': session[1], 
            'student_name': session[2],
            'start_time': session[3],
            'end_time': session[4],
            'total_alerts': session[5],
            'status': session[6]
        })
    
    return session_list

@app.get("/api/session/{session_id}/results")
async def get_session_results(session_id: str):
    """Get detection results for specific session"""
    conn = sqlite3.connect('proctoring_results.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM detection_results 
        WHERE session_id = ? 
        ORDER BY timestamp DESC
    ''', (session_id,))
    results = cursor.fetchall()
    conn.close()
    
    result_list = []
    for result in results:
        result_list.append({
            'timestamp': result[2],
            'result_type': result[3],
            'result_value': result[4],
            'alert_count': result[5]
        })
    
    return result_list

@app.post("/api/upload-student-photo")
async def upload_student_photo(file: UploadFile = File(...), student_name: str = ""):
    """Upload student photo to student_db folder"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_path = f"student_db/{student_name}.jpg"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return {
        'message': f'Photo uploaded successfully for {student_name}',
        'file_path': file_path,
        'note': 'Restart API to reload face encodings'
    }

@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(proctoring_sessions.active_sessions),
        'registered_students': len(known_face_names),
        'student_names': known_face_names,
        'message': 'Cheating Detection API is running with your existing models!'
    }

@app.post("/api/test-detection")
async def test_detection_simple():
    """Simple detection test for Postman"""
    
    # Simulate your actual detection flow
    results = []
    alert_count = 0
    
    # Test scenarios
    test_scenarios = [
        "result:[SAFE]",
        "result:[SAFE]", 
        "result:[ALERT-Unknown Face] (1)",
        "result:[SAFE]",
        "result:[ALERT-Banned Objects Detected] (2)",
        "result:[SAFE]",
        "result:[ALERT-Looking Away - Head Pose] (3)",
        "result:[SAFE]",
        "result:[ALERT-Mouth Open] (4)",
        "result:[ALERT-High Audio Level] (5)"
    ]
    
    for i, scenario in enumerate(test_scenarios):
        results.append(scenario)
        if "ALERT" in scenario:
            alert_count = int(scenario.split('(')[1].split(')')[0])
            
        # Also add result2 simulation
        if i % 3 == 0:
            results.append("result2:[]")
        else:
            results.append("result2:[]")
            
        if alert_count >= 5:
            results.append("MAXIMUM ALERTS REACHED (5)! STOPPING PROCTORING SYSTEM!")
            break
    
    return {
        'test_results': results,
        'total_alerts': alert_count,
        'session_terminated': alert_count >= 5,
        'message': 'This is a simulation of your detection system'
    }

@app.websocket("/ws/monitor/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Real-time monitoring via WebSocket"""
    await websocket.accept()
    logger.info(f"WebSocket connected for session: {session_id}")
    
    processor = DetectionProcessor()
    
    try:
        while processor.session_active:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'frame':
                results = processor.process_frame(message['data'], session_id)
                await websocket.send_text(json.dumps(results))
                
                if not processor.session_active:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Cheating Detection API with your existing models...")
    print("📊 Dashboard: http://localhost:8000")
    print("📋 API Docs: http://localhost:8000/docs")
    print(f"👥 Registered Students: {len(known_face_names)}")
    uvicorn.run(app, host="127.0.0.1", port=8000)