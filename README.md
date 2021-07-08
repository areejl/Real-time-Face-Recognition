# Real-time-Face-Recognition
## Real-time Face Recognition by OpenCV & python - task 4 for track AI at Smart-Methods summer training

>*Note: You should have Python3.8 and Visual studio code installed first.*

1. [install Python 3.8.3](https://www.python.org/downloads/release/python-383/) 
**Check Add Python 3.8 to PATH**
2. Downloaded [Visual Studio](https://visualstudio.microsoft.com/downloads/) with Python Development package, Node.JS, Desktop web c++
3. Open windows Powershell
```
$ cd .\Desktop\
$ mkdir facerecog
$ cd .\facerecog\
$ git clone git://github.com/ageitgey/face_recognition
$ pip install virtualenv
$ virtualevn env
```
> *Note : "env" is the name of your virtualevn*

**activite the virtualevn**
```
$ cd .\env\
$ cd .\Scripts\
$ cd .\activate\
```
> Error : PowerShell says “execution of scripts is disabled on this system.” when I run activate 

Solution :
As an Administrator, you can set the execution policy by typing this into your PowerShell window:

$ Set-ExecutionPolicy RemoteSigned
For more information, see Using the Set-ExecutionPolicy Cmdlet.

When you are done, you can set the policy back to its default value with:

$ Set-ExecutionPolicy Restricted
You may see an error:

Access to the registry key
'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell' is denied. 
To change the execution policy for the default (LocalMachine) scope, 
  start Windows PowerShell with the "Run as administrator" option. 
To change the execution policy for the current user,  run "Set-ExecutionPolicy -Scope CurrentUser".

So you may need to run the command like this :
$ Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```
$ cd..
$ cd..
$ ls
$ cd .\face_recogintion\
$ python setup.py install
$ pip install cmake
$ python setup.py install
```
Now you can run $ pip freeze to check that all dep installed

4. Put photo in face_recogintion dirctory 
the face in photo it should be clear 
5. Open face_recogintion folder in visual studio 
6. create python file "fileName.py"
7. run the following code 
```
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

```
### Results :
*Unknown**</br>
![unknown](unknown.png)</br>
*known**</br>
![levi](levi.png)
