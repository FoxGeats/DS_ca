# Facial Recognition Live Streaming Application


This application uses Flask and Socket io to create a server and waitress to allow multi threading og the application.

It uses the face_recognition library along with open cv to perform facial recognition on the faces in the faceImages folder.


To sart the application:

Pip3 installs
 
```
pip3 install requirements.txt

```

In most cases, it doesnt work so you might want to try installing them individually.

Almost everyone failed when install dlib, you need to install C++ environment first, and install cmake and add it to the system path, then install dlib. You should also at least use python 3.9 or later.


To Launch

```
run the file: server.py

```
Once the server script is ran successfuly you can navigate to http://localhost:8080 or the link that is given by the console.

Video : https://youtu.be/LU8XtTWtjbY