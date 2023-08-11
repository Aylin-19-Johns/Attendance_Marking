from flask import Flask, render_template, request, redirect, session
import cv2
import numpy as np
import face_recognition
import os
import datetime
import threading
import csv

start_q = True
processed_faces = set()


def process_attendance():
    if not os.path.exists('AttendanceSheets'):
        os.makedirs('AttendanceSheets')

    path = 'STUDENTS'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    def findEncodings(images):
        if len(images) == 0:
            return []
        else:
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList

    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    while start_q:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)

                if name not in processed_faces:  # Check if face has already been processed
                    processed_faces.add(name)  # Add the name to the set of processed faces

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), font, 1, (255, 255, 255), 2)

                    now = datetime.datetime.now()
                    date = now.strftime('%d-%m-%Y')
                    time = now.strftime('%H:%M:%S')

                    file_name = date + '.csv'
                    file_path = os.path.join('AttendanceSheets', file_name)

                    if not os.path.isfile(file_path):
                        with open(file_path, 'w') as f:
                            f.write('Name, Date, Time\n')

                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if row[0] == name:  # Check if name already exists in the file
                                break
                        else:  # Executes if no break occurs in the loop
                            with open(file_path, 'a') as f:
                                f.write(f'{name}, {date}, {time}\n')

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


app = Flask(__name__)
attendance_thread = None
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Define the admin username and password
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password'

def start_service():
    global start_q
    start_q = True

    global attendance_thread
    attendance_thread = threading.Thread(target=process_attendance)
    attendance_thread.start()


def stop_service():
    global start_q
    start_q = False

    global attendance_thread
    if attendance_thread is not None:
        attendance_thread.join()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start")
def start():
    start_service()
    return redirect("/")

@app.route("/stop")
def stop():
    stop_service()
    return redirect("/")

@app.route("/upload", methods=["POST"])
def upload():
    # Get the image file from the request
    image = request.files["image"]
    name = request.form["name"]

    # Save the image file to a directory
    image.save("STUDENTS/" + name + ".jpg")

    stop_service()
    start_service()
    # Redirect the user back to the index page
    return redirect("/")

@app.route("/admin")
def admin():
    if 'admin' in session:
        return render_template("admin.html")
    else:
        return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect("/admin")
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html", error=None)

@app.route("/logout")
def logout():
    session.pop('admin', None)
    return redirect("/login")



if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)