Face Recognition Attendance System
A complete Face Recognition–based Attendance Management System built using Python, OpenCV, face_recognition, Flask, and MongoDB.
This system captures faces in real time, recognizes registered students, and automatically marks attendance with timestamps and session control.

Features:
Face Recognition

1.Real-time face detection using OpenCV.
2.High-accuracy face recognition using face_recognition library.
3.Supports encoding and storing multiple student faces.

Attendance Management

1.Automatic attendance marking when a face is detected.
2.Session-wise attendance (Morning/Evening or custom sessions).
3.Prevents duplicate attendance within the same session.
4.Stores timestamp and date of attendance.

Student Management

1.Add students with name, roll number, department, and face encoding.
2.Update & delete student information easily.
3.Stores face encodings using pickle.

Database
Uses MongoDB via Flask-PyMongo
Collections:
students
attendance
users (for login)

Admin Authentication

1.Secure login system using hashed passwords
2.Only logged-in users can access admin pages

Frontend
1.Built using Flask templates (HTML + CSS)
2.User-friendly web interface for:
->Live camera stream
->Attendance dashboard
->Student list
->Add/Delete students
->Export attendance
