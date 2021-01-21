#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>

using namespace std;
using namespace cv;

void measureDist();
bool validArea(Point faceCentre);
void measureWidth(Point faceCentre);
void measureHeight(Point faceCentre);

//String face_cascade_file = "haarcascade_frontalface_alt.xml";
String face_cascade_file = "lbpcascade_frontalface.xml";
int countConst = 2;
int cutoffL, cutoffR, cutoffT, cutoffB = 0;
float fovH = 64, fovV = 34;
float screenWidth = 0.374, screenHeight = 0.344;
float distanceScene = 1;
float faceWdM = 0.205;

CascadeClassifier face_cascade;
String window_name = "Optical Cloaking Device";
std::vector<Rect> faces;

int frameFWdP, frameFHtP;
int frameBWdP, frameBHtP;
int faceWdP, faceHtP;
float frameWidth, frameHeight;
float proportion;
float distancePerson;
int fCount;

int main()
{

	VideoCapture captureF, captureB;
	Mat frameF, frameB, frameGrey, prevGrey, cutFrame;
    vector<Point2f> points[2];
    vector<Rect> faces;

	if (!face_cascade.load(face_cascade_file)) {
		cout << "Error loading face cascade" << endl;
		return -1;
	};

	captureF.open(0);
	captureB.open(1);

	while (captureF.read(frameF) && captureB.read(frameB)) {

		frameFWdP = frameF.cols;
		frameFHtP = frameF.rows;
		frameBWdP = frameB.cols;
		frameBHtP = frameB.rows;

        cvtColor(frameF, frameGrey, COLOR_BGR2GRAY);
        imshow("Greyscale", frameGrey);
        equalizeHist(frameGrey, frameGrey);
        imshow("equalizeHist", frameGrey);

        //face_cascade.detectMultiScale( frameGrey, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(80, 80) );
        face_cascade.detectMultiScale(frameGrey, faces, 1.1, 2, 0, Size(80, 80));

        if( faces.size() > 0 ) {

        		Mat maskFace(frameGrey.size(), CV_8UC1, Scalar::all(0));
        		maskFace(faces[0]).setTo(Scalar::all(255));
        		goodFeaturesToTrack(frameGrey, points[1], 25, 0.01, 10, maskFace, 3, 0, 0.04);

        } else if( !points[0].empty() ) {
        	//finds new points
            vector<uchar> status;
            vector<float> err;
            TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);

            calcOpticalFlowPyrLK(prevGrey, frameGrey, points[0], points[1], status, err, Size(30,30), 3, termcrit, 0, 0.001);

            //removes points that werent found and places points on frame
            size_t i, k;

            for( i = k = 0; i < points[1].size(); i++ ) {

                if( !status[i] ) {
                    continue;
                }

                points[1][k++] = points[1][i];
                circle( frameF, points[1][i], 3, Scalar(0,255,0), -1, 8);

            }

            points[1].resize(k);
            faces.push_back(boundingRect(points[1]));
        }

			faceWdP = faces[0].width;
			faceHtP = faces[0].height;

			//Finds point centre of face
			Point faceCentre( faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5 );

			//Calculate the distance of the detected person
			measureDist();

			if(validArea(faceCentre)) {

				measureWidth(faceCentre);
				measureHeight(faceCentre);

			} else {

				cout << "Person is Outside Valid Area" << endl;

			}

			if(!faces.empty()){
				//Draw rect around face
				rectangle(frameF, faces[0], 255, 1, 8, 0);

				//Cut edges from frame
				if (frameBWdP > frameBWdP - cutoffL - cutoffR) {

					cutFrame = frameB(Rect(cutoffL, cutoffT, frameBWdP - cutoffL - cutoffR, (frameBHtP - cutoffT - cutoffB))).clone();

				}
			}


        std::swap(points[1], points[0]);
        cv::swap(prevGrey, frameGrey);

        imshow("FaceDetection", frameF);

		if (cutFrame.empty()) {

			resize(frameF, frameF, Size(320, 180));
			resize(frameB, frameB, Size(1366, 768));

			frameF.copyTo(frameB(Rect(0, 0, frameF.cols, frameF.rows)));
			imshow(window_name, frameB);

		} else {

			resize(frameF, frameF, Size(320, 180));
			resize(cutFrame, cutFrame, Size(1366, 768));

			frameF.copyTo(cutFrame(Rect(0, 0, frameF.cols, frameF.rows)));
			imshow(window_name, cutFrame);
		}




		int key = waitKey(10);

		if ((char) key == 27) {

			break;

		}

}

    return 0;

}





// MEASURING DISTANCE
void measureDist() {

	//proportion of screen taken up by face (pixels)
	proportion = (float) faceWdP / (float) frameFWdP;
	//finds the width of the front camera frame (metres)
	frameWidth = faceWdM / proportion;
	//Calculates distance of person
	distancePerson = (frameWidth / 2) / tan((fovH/2)*(3.14159265/180));

}

//CHECK IF PERSON IS WITHIN WORKING AREA
bool validArea(Point faceCentre) {

	float minDistance, theta, validRegionWidth, proportion, validRegionWidthP;

	float widthScene = 2 * distanceScene * tan( (fovH/2) * (3.14159265/180) );

	//Calculates the minimum distance a person can be (m)
	minDistance = ((screenWidth/2) * distanceScene) / ((widthScene - screenWidth)/2);

	if (distancePerson >= minDistance) {

		theta = atan((screenWidth/2) / minDistance);
		//Half of valid width (m)
		validRegionWidth = (distancePerson - minDistance) * tan(theta);
		//Half of proportion of image taken by valid region
		proportion = validRegionWidth / frameWidth;
		//Half of valid width (p)
		validRegionWidthP = frameFWdP * proportion;

		if((faceCentre.x <= ((frameFWdP / 2) + validRegionWidthP)) && (faceCentre.x >= ((frameFWdP / 2) - validRegionWidthP))) {

			return true;

		}

	} else {

		cout << "Too close" << endl;

	}

	return false;

}

void measureWidth(Point faceCentre) {

	float side0, side1, side3, side4;
	float theta0, theta1;
	float frameLM, frameRM;
	float propL, propR;

	//Calculate persons distance from centre (p)
	float offsetP = ((float) frameFWdP / 2) - (float) faceCentre.x;
	//Calculate the width of the scene (m)
	float widthScene = 2 * distanceScene * tan( (fovH/2) * (M_PI/180) );
	//Distance between person and scene (m)
	float distanceTotal = distancePerson + distanceScene;
	//Proportion of frame taken up by offset
	proportion = (float) abs(offsetP) / (float) frameFWdP;
	//Offset from centre (m)
	float offsetM = proportion * frameWidth;

	//If within edges of screen
	if (offsetM < 0.187) {

		//If in middle
		if (offsetP == 0) {

			//Calculate angle
			float theta = atan((screenWidth / 2) / distancePerson);
			//Half of blocked portion (m)
			float side0 = distanceTotal * tan(theta);
			//How much to cut off (m)
			frameLM = (widthScene / 2) - side0;
			//How much of frame to cut from each side (%)
			proportion = frameLM / widthScene;
			//Cut off (p)
			cutoffL = (int) (frameBWdP * proportion);
			//Equal on each side
			cutoffR = cutoffL;

			return;

		}

		//Left (0) and right (1) side of the screen (m)
		side0 = (screenWidth / 2) + offsetM;
		side1 = screenWidth - side0;

		//Angles for left (0) and right (1) triangle
		theta0 = atan(side0 / distancePerson);
		theta1 = atan(side1 / distancePerson);

		//Left(4), right(3) side of blocked frame
		side4 = distanceTotal * tan(theta0);
		side3 = distanceTotal * tan(theta1);

		//Person moves towards the right
		if (offsetP > 0) {

			//Left and right portions of frame (m)
			frameLM = (widthScene / 2) + offsetM - side4;
			frameRM = (widthScene / 2) - offsetM - side3;

		//Person moves towards the left
		} else if (offsetP < 0) {

			//left and right to cut off (m)
			frameLM = (widthScene / 2) - offsetM - side3;
			frameRM = (widthScene / 2) + offsetM - side4;

		}

		//(%)
		propL = frameLM / widthScene;
		propR = frameRM / widthScene;

		//(p)
		cutoffL = (int) (propL * frameBWdP);
		cutoffR = (int) (propR * frameBWdP);

		return;

		//If outside edges of screen
	} else if (offsetM > 0.187){

		//Left or right side on screen (m)
		//smaller side (0), larger side (1)
		side0 = offsetM - 0.187;
		side1 = side0 + screenWidth;

		//Angles of above
		theta0 = atan(side0 / distancePerson);
		theta1 = atan(side1 / distancePerson);

		//left and right side at scene (m)
		//larger (4), smaller (3)
		side3 = distanceTotal * tan(theta0);
		side4 = distanceTotal * tan(theta1);

		//Person moves towards the right
		if (offsetP > 0) {

			//(m)
			frameRM = ((widthScene / 2) - offsetM) + side3;
			frameLM = widthScene - frameRM - side4;

			if (frameLM < 0) {

				frameLM = 0;

			}

			//Person moves towards the left
		} else if (offsetP < 0){

			//(m)
			frameLM = ((widthScene / 2) - offsetM) + side3;
			frameRM = widthScene - frameLM - side4;

			if (frameRM < 0) {

				frameRM = 0;

			}

		}

		propL = frameLM / widthScene;
		propR = frameRM / widthScene;

		cutoffL = (int) (propL * frameBWdP);
		cutoffR = (int) (propR * frameBWdP);
		return;
	}

}

void measureHeight(Point faceCentre) {

	float frameTM;

	//offset (p)
	float offsetP = ((float)frameFHtP/2) - (float)faceCentre.y;
	//half of height of scene (m)
	float heightScene = distanceScene * tan((fovV/2)* (3.14159265/180));
	//height of front frame (m)
	float frameHeight = distancePerson * tan((fovV/2)* (3.14159265/180));
	//proportion of frame taken up by offset
	proportion = (float) abs(offsetP) / (float) frameFHtP;
	//offset (m)
	float offsetM = proportion * frameHeight;
	//angle
	float theta = atan(offsetM / distancePerson);
	//portion of image between centre and LoS (m)
	float side = distanceScene * tan(theta);

	if (offsetP == 0) {

		cutoffT = frameBHtP / 2;
		cutoffB = 0;//(int) (propB * frameHtP);
		return;

		//above screen
	} else if (offsetP > 0) {

		frameTM = (heightScene /2) + side;

		//below screen
	} else if (offsetP < 0) {

		frameTM = (heightScene / 2) - side;

	}

	float propT = frameTM / heightScene;

	cutoffT = (int) (propT * frameBHtP);

}
