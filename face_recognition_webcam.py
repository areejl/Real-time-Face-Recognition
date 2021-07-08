import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0)

levi_image = fr.load_image_file("levi.jpg")
levi_face_encoding = fr.face_encodings(levi_image)[0]#Analysis the picture (eyes,nose,size) then encode it 
known_face_encondings = [levi_face_encoding]

known_face_names = ["Levi"]

while True: 
    ret, frame = video_capture.read()#

    rgb_frame = frame[:, :, ::-1]#change the color to rgb frame

    face_locations = fr.face_locations(rgb_frame)#where are the faces in the frame it coulde be many faces 
    face_encodings = fr.face_encodings(rgb_frame, face_locations)#which face in the frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()