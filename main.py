import cv2
import numpy as np
import face_recognition
import os
import datetime

def process_attendance():
    start_q = True

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

    processed_faces = []  # List to track processed faces

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

                    with open(file_path, 'a') as f:
                        f.write(f'{name}, {date}, {time}\n')

                    processed_faces.append(name)  # Add the processed face to the list

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
