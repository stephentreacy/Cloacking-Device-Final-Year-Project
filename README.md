# Cloacking-Device-Final-Year-Project
Final year project for my degree in 2015.

The idea is to allow a person to see through something, for example, a wall. If a person is standing in front of a wall and a ball rolls behind it, the ball is blocked from the person’s view. If a screen is placed on the wall and a camera is placed behind the wall then the person is able to view the ball rolling but the exact location of the ball is unknown to the person as the image is dependant of the location of the camera. If a second camera and a computer capable of running image processing tasks is added to detect the person and calculate their location relative to the screen and wall then the image shown on the screen may be changed to give the viewer the impression that they are looking through the wall with the ball being in the same position on the screen as it is behind the wall. Using this type of method blind spots in locations such as the pillars of a car can be removed, improving the visibility of the driver.

![alt text](https://github.com/stephentreacy/Cloacking-Device-Final-Year-Project/blob/main/images/ball_intro.png?raw=true)

The figure above illustrates the example of the ball rolling behind a wall. At the left the ball is partially behind the wall and partially visible to the person, the portion not visible to the person is shown on the left side of the screen. The centre images show the ball completely behind the wall, the ball is fully shown on the screen. Finally the ball is on the right side of the wall, partially visible just as the image on the left except this time the part of the ball which is not seen is shown on the right.

The project comprises of a laptop using Open Source Computer Vision (OpenCV) with C++ and a second camera. OpenCV is a widely used library for computer vision and machine learning. Many functions to complete common image processing tasks such as face detection and object tracking are included with OpenCV. Frames from the laptop’s integrated camera are used by OpenCV to detect the person’s face. The second webcam faces behind the laptop screen and the video stream is show on the screen. The main parts of the project are face detection, optical flow, calculating the distance of the person and if they are in a valid area, and rendering the image.


