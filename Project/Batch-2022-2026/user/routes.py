import cv2
import os
import datetime
from flask import render_template, Response, request, redirect, url_for, flash
from user import user_bp
from database import get_connection
from face_utils_recog import recognize_faces
from validate import preprocess

# Global variables for camera control
camera = None
attendance_marked = set()


@user_bp.route("/attendance")
def attendance():
    if preprocess()=="valid":
        """Render webcam attendance page"""
        return render_template("attendance.html")
    else:
        return render_template("base.html")


def gen_frames():
    """Generate webcam frames with face recognition"""
    global camera, attendance_marked
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Recognize faces using the trained models
        recognized_name = recognize_faces(frame)

        # If a recognized face, mark attendance
        if recognized_name and recognized_name not in attendance_marked:
            mark_attendance(recognized_name)
            attendance_marked.add(recognized_name)

        # Show recognition on frame
        if recognized_name:
            cv2.putText(frame, f"{recognized_name}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@user_bp.route('/video_feed')
def video_feed():
    """Video stream route"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def mark_attendance(student_name):
    """Store attendance in database"""
    conn = get_connection()
    cursor = conn.cursor()
    date = datetime.date.today()
    time = datetime.datetime.now().strftime("%H:%M:%S")

    cursor.execute(
        "INSERT INTO attendance (student_name, date, time) VALUES (?, ?, ?)",
        (student_name, date, time)
    )

    conn.commit()
    conn.close()
    print(f"[INFO] Attendance marked for {student_name} at {time}")


@user_bp.route('/report', methods=['GET', 'POST'])
def report():
    """Show today's attendance report"""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.date.today()

    cursor.execute("SELECT student_name, time FROM attendance WHERE date=?", (today,))
    records = cursor.fetchall()
    conn.close()

    return render_template("report.html", records=records, today=today)


@user_bp.route('/stop_camera')
def stop_camera():
    """Stop webcam feed"""
    global camera
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()
    return redirect(url_for('user.attendance'))
