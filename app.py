import os
import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime, time
import pandas as pd
from flask import Flask, render_template, Response, request, redirect, url_for, flash, send_file, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey123'  # Change this in production

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/attendance_db"
mongo = PyMongo(app)

# Camera configuration
CAMERA_SOURCE = 0  # Default is 0 for primary camera, change if needed

# Time slots configuration
MORNING_START = time(9, 30)
MORNING_END = time(12, 30)
AFTERNOON_START = time(14, 0)
AFTERNOON_END = time(16, 30)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize attendance system
def init_attendance_system():
    # Load trained model or create empty lists if file doesn't exist
    try:
        with open("face_encodings.pkl", "rb") as f:
            model_data = pickle.load(f)
        encodings = model_data.get("encodings", [])
        names = model_data.get("names", [])
        roll_numbers = model_data.get("roll_numbers", [])
    except (FileNotFoundError, EOFError):
        encodings = []
        names = []
        roll_numbers = []
    
    # Ensure directories exist
    os.makedirs("static/captures", exist_ok=True)
    
    # Initialize attendance files if they don't exist
    if not os.path.exists("attendance.xlsx"):
        pd.DataFrame(columns=['roll_number', 'name', 'date', 'time', 'status', 'image_path', 'session']).to_excel("attendance.xlsx", index=False)
    
    return encodings, names, roll_numbers

known_face_encodings, known_face_names, known_face_roll_numbers = init_attendance_system()
attendance_records = []

def load_attendance():
    global attendance_records
    try:
        if os.path.exists("attendance.xlsx"):
            df = pd.read_excel("attendance.xlsx")
            # Ensure proper data types
            df['roll_number'] = df['roll_number'].astype(str).str.strip()
            df['name'] = df['name'].astype(str).str.strip()
            attendance_records = df.to_dict('records')
        else:
            attendance_records = []
    except Exception as e:
        print(f"Error loading attendance: {e}")
        attendance_records = []

def save_attendance_to_excel():
    try:
        df = pd.DataFrame(attendance_records)
        # Ensure proper data types before saving
        df['roll_number'] = df['roll_number'].astype(str).str.strip()
        df['name'] = df['name'].astype(str).str.strip()
        df.to_excel("attendance.xlsx", index=False)
    except Exception as e:
        print(f"Error saving attendance: {e}")
        flash('Error saving attendance data', 'danger')

def save_attendance_to_csv():
    try:
        df = pd.DataFrame(attendance_records)
        # Ensure proper data types before saving
        df['roll_number'] = df['roll_number'].astype(str).str.strip()
        df['name'] = df['name'].astype(str).str.strip()
        df.to_csv("attendance.csv", index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")
        flash('Error saving CSV data', 'danger')

# Load existing attendance at startup
load_attendance()

def validate_roll_number(roll_number):
    """Validate and clean roll number input"""
    return str(roll_number).strip()

def validate_name(name):
    """Validate and clean name input"""
    import re
    return re.sub(r'[^a-zA-Z\s]', '', str(name).strip())

def is_within_time_slots(current_time):
    """Check if current time is within allowed time slots"""
    current_time = current_time.time()
    return ((MORNING_START <= current_time <= MORNING_END) or 
            (AFTERNOON_START <= current_time <= AFTERNOON_END))

def can_mark_attendance(roll_number, date_str, session):
    """Check if attendance can be marked (1 hour gap between sessions)"""
    # Get all records for this student on this date
    student_records = [r for r in attendance_records 
                      if str(r['roll_number']).strip() == roll_number 
                      and r['date'] == date_str]
    
    if not student_records:
        return True
    
    # Check if same session already marked
    if any(r['session'] == session for r in student_records):
        return False
    
    # For different session, check time gap
    last_record = max(student_records, key=lambda x: datetime.strptime(x['time'], '%H:%M:%S'))
    last_time = datetime.strptime(last_record['time'], '%H:%M:%S').time()
    current_time = datetime.now().time()
    
    # Calculate time difference in hours
    time_diff = (datetime.combine(datetime.today(), current_time) - 
                datetime.combine(datetime.today(), last_time)).total_seconds() / 3600
    
    return time_diff >= 1

def mark_attendance(roll_number, name, frame):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Determine session (morning or afternoon)
    current_time = now.time()
    if MORNING_START <= current_time <= MORNING_END:
        session = 'morning'
    elif AFTERNOON_START <= current_time <= AFTERNOON_END:
        session = 'afternoon'
    else:
        session = 'invalid'
    
    try:
        # Clean and validate inputs
        roll_number = validate_roll_number(roll_number)
        name = validate_name(name)
        
        # Check if within allowed time slots
        if not is_within_time_slots(now):
            flash(f'Attendance can only be marked between {MORNING_START.strftime("%H:%M")}-{MORNING_END.strftime("%H:%M")} and {AFTERNOON_START.strftime("%H:%M")}-{AFTERNOON_END.strftime("%H:%M")}', 'warning')
            return False
        
        # Check if attendance can be marked (1 hour gap)
        if not can_mark_attendance(roll_number, date_str, session):
            flash(f'Attendance already marked for this session or minimum 1 hour gap required', 'warning')
            return False
        
        # Save captured image
        image_filename = f"{roll_number}_{date_str}_{time_str.replace(':', '-')}.jpg"
        image_path = os.path.join("static/captures", image_filename)
        
        # Convert frame to RGB before saving
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, rgb_frame)

        attendance_records.append({
            'roll_number': roll_number,
            'name': name,
            'date': date_str,
            'time': time_str,
            'status': 'Present',
            'image_path': f"captures/{image_filename}",
            'session': session
        })

        save_attendance_to_excel()
        save_attendance_to_csv()
        return True
        
    except Exception as e:
        print(f"Error marking attendance: {e}")
        flash('Error marking attendance', 'danger')
        return False

def generate_frames():
    camera = cv2.VideoCapture(CAMERA_SOURCE)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        face_roll_numbers = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            roll_number = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                roll_number = validate_roll_number(known_face_roll_numbers[best_match_index])
                name = validate_name(known_face_names[best_match_index])
                mark_attendance(roll_number, name, frame.copy())

            face_names.append(name)
            face_roll_numbers.append(roll_number)

        # Draw boxes and labels
        for (top, right, bottom, left), name, roll_number in zip(face_locations, face_names, face_roll_numbers):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name} ({roll_number})", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Add timestamp and time slot info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = datetime.now().time()
        
        if MORNING_START <= current_time <= MORNING_END:
            time_slot = "Morning (9:30-12:30)"
        elif AFTERNOON_START <= current_time <= AFTERNOON_END:
            time_slot = "Afternoon (2:00-4:30)"
        else:
            time_slot = "Outside allowed time slots"
        
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, time_slot, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('Username and password are required', 'danger')
            return redirect(url_for('register'))
        
        if mongo.db.users.find_one({'username': username}):
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({
            'username': username,
            'password': hashed_password,
            'created_at': datetime.now()
        })
        
        flash('Registration successful! Please log in', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = mongo.db.users.find_one({'username': username})
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# Main application routes
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

from datetime import datetime

@app.route('/dashboard')
@login_required
def dashboard():
    # Get filter parameters from request
    roll_number = request.args.get('roll_number', '').strip()
    name = request.args.get('name', '').strip()
    from_date = request.args.get('from_date', '')
    to_date = request.args.get('to_date', '')
    
    filtered_records = attendance_records
    
    # Apply filters
    if roll_number:
        filtered_records = [r for r in filtered_records if roll_number.lower() in str(r['roll_number']).lower()]
    
    if name:
        filtered_records = [r for r in filtered_records if name.lower() in r['name'].lower()]
    
    if from_date:
        filtered_records = [r for r in filtered_records if r['date'] >= from_date]
    
    if to_date:
        filtered_records = [r for r in filtered_records if r['date'] <= to_date]
    
    sorted_records = sorted(filtered_records, 
                          key=lambda x: (x['date'], x['time']), 
                          reverse=True)
    
    # Get unique dates for date filter dropdown
    unique_dates = sorted(set(r['date'] for r in attendance_records), reverse=True)
    
    # Get current time for display
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('dashboard.html', 
                         attendance=sorted_records[:100],
                         roll_number=roll_number,
                         name=name,
                         from_date=from_date,
                         to_date=to_date,
                         unique_dates=unique_dates,
                         current_time=current_time)
@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_attendance')
@login_required
def take_attendance():
    return render_template('attendance.html')

@app.route('/download_attendance')
@login_required
def download_attendance():
    save_attendance_to_excel()
    return send_file("attendance.xlsx", as_attachment=True)

@app.route('/download_csv')
@login_required
def download_csv():
    save_attendance_to_csv()
    return send_file("attendance.csv", as_attachment=True)

@app.route('/delete_record/<int:record_id>', methods=['POST'])
@login_required
def delete_record(record_id):
    if 0 <= record_id < len(attendance_records):
        record = attendance_records[record_id]
        if 'image_path' in record and record['image_path']:
            try:
                image_path = os.path.join('static', record['image_path'])
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"Error deleting image: {e}")
        
        attendance_records.pop(record_id)
        save_attendance_to_excel()
        save_attendance_to_csv()
        flash('Record deleted successfully!', 'success')
    else:
        flash('Invalid record ID', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/toggle_status/<int:record_id>', methods=['POST'])
@login_required
def toggle_status(record_id):
    if 0 <= record_id < len(attendance_records):
        record = attendance_records[record_id]
        record['status'] = 'Present' if record['status'] == 'Absent' else 'Absent'
        save_attendance_to_excel()
        save_attendance_to_csv()
        flash('Status updated successfully!', 'success')
    else:
        flash('Invalid record ID', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/manual_entry', methods=['POST'])
@login_required
def manual_entry():
    try:
        roll_number = validate_roll_number(request.form['roll_number'])
        name = validate_name(request.form['name'])
        status = request.form['status']
        date = request.form['date']
        time = request.form['time']
        
        # Determine session
        entry_time = datetime.strptime(time, '%H:%M').time()
        if MORNING_START <= entry_time <= MORNING_END:
            session = 'morning'
        elif AFTERNOON_START <= entry_time <= AFTERNOON_END:
            session = 'afternoon'
        else:
            session = 'invalid'
        
        existing = next((record for record in attendance_records 
                       if str(record['roll_number']).strip() == roll_number 
                       and record['date'] == date
                       and record['session'] == session), None)
        
        if existing:
            flash('Attendance for this student on this date and session already exists', 'warning')
        else:
            attendance_records.append({
                'roll_number': roll_number,
                'name': name,
                'date': date,
                'time': time + ':00',  # Add seconds for consistency
                'status': status,
                'image_path': None,
                'session': session
            })
            save_attendance_to_excel()
            save_attendance_to_csv()
            flash('Manual attendance entry added successfully!', 'success')
    except Exception as e:
        flash(f'Error adding manual entry: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
