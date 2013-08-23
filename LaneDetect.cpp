/*------------------------------------------------------------------------------------------*\
   Lane Detection

   General idea and some code modified from:
   chapter 7 of Computer Vision Programming using the OpenCV Library. 
   by Robert Laganiere, Packt Publishing, 2011.

   This program is free software; permission is hereby granted to use, copy, modify, 
   and distribute this source code, or portions thereof, for any purpose, without fee, 
   subject to the restriction that the copyright notice may not be removed 
   or altered from any source or altered source distribution. 
   The software is released on an as-is basis and without any warranties of any kind. 
   In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
   The author disclaims all warranties with regard to this software, any use, 
   and any consequent failure, is purely the responsibility of the user.
 
   Copyright (C) 2013 Jason Dorweiler, www.transistor.io


Notes: 

	Can we compare the Hough and HoughP images? 
	eg: Bitwise and on Hough and contour image
	    Bitwise and on HoughP and result of previous image. need to write HoughP to blank Mat of same size
	    Filter for angles based on theta, there should be a ~pi/2 angle on the otherside
	    of the road.  If there isn't one then drop that line for the list. 

	Add up number on lines that are found within a threshold of a given rho,theta and 
	use that to determine a score.  Only lines with a good enough score are kept. 

	Calculation for the distance of the car from the center.  This should also determine
	if the road in turning.  We might not want to be in the center of the road for a turn. 

	The minimum vote used in the Hough filter is very sensitive to the image used.  It might be a 
	good idea to set the algo up to reprocess the image if the bitwise adding of the two hough
	filters does not produce at least two lines.
	
	Several other parameters can be played with: min vote on houghp, line distance and gap.  Some
	type of feed back loop might be good to self tune these parameters. 

	We are still finding the Road, i.e. both left and right lanes.  we Need to set it up to find the
	yellow divider line in the middle. 
\*------------------------------------------------------------------------------------------*/

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "linefinder.h"
//#include "edgedetector.h"

#define PI 3.1415926

using namespace cv;

int main(int argc, char* argv[]) {
	int houghVote = 200;
	string arg = argv[1];
	bool showSteps = argv[2];

	string window_name = "Processed Video";
	namedWindow(window_name, CV_WINDOW_KEEPRATIO); //resizable window;
	//VideoCapture capture(arg);

	//if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        //	{capture.open(atoi(arg.c_str()));}

        Mat image=imread(argv[1]);
        while (1)
        {/*
            //	capture >> image;
            if (image.empty())
                break;
            	Mat gray;
            	cvtColor(image,gray,CV_RGB2GRAY);
            	vector<string> codes;
            	Mat corners;
            	findDataMatrix(gray, codes, corners);
            	drawDataMatrixCodes(image, codes, corners);*/

    // Display the image
	if(showSteps){
		namedWindow("Original Image");
		imshow("Original Image",image);
		imwrite("original.bmp", image);
	}

   // Canny algorithm
	Mat contours;
	Canny(image,contours,50,350);
	Mat contoursInv;
	threshold(contours,contoursInv,128,255,THRESH_BINARY_INV);

   // Display Canny image
	if(showSteps){
		namedWindow("Contours");
		imshow("Contours",contoursInv);
		imwrite("contours.bmp", contoursInv);
	}

 /* 
	Hough tranform for line detection with feedback
	Increase by 25 for the next frame if we found some lines.  
	This is so we don't miss other lines that may crop up in the next frame
	but at the same time we don't want to start the feed back loop from scratch. 
*/
	std::vector<Vec2f> lines;
	if (houghVote < 1 or lines.size() > 2){ // we lost all lines. reset 
		houghVote = 200; 
	}
	else{ houghVote += 25;} 
	while(lines.size() < 5 && houghVote > 0){
		HoughLines(contours,lines,1,PI/180, houghVote);
		houghVote -= 5;
	}
	std::cout << houghVote << "\n";
	Mat result(contours.rows,contours.cols,CV_8U,Scalar(255));
	image.copyTo(result);

   // Draw the limes
	std::vector<Vec2f>::const_iterator it= lines.begin();
	Mat hough(image.size(),CV_8U,Scalar(0));
	while (it!=lines.end()) {

		float rho= (*it)[0];   // first element is distance rho
		float theta= (*it)[1]; // second element is angle theta
		
		//if (theta < PI/20. || theta > 19.*PI/20.) { // filter theta angle to find lines with theta between 30 and 150 degrees (mostly vertical)
		
			// point of intersection of the line with first row
			Point pt1(rho/cos(theta),0);        
			// point of intersection of the line with last row
			Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
			// draw a white line
			line( result, pt1, pt2, Scalar(255), 8); 
			line( hough, pt1, pt2, Scalar(255), 8);
		//}

		//std::cout << "line: (" << rho << "," << theta << ")\n"; 
		++it;
	}

    // Display the detected line image
	if(showSteps){
		namedWindow("Detected Lines with Hough");
		imshow("Detected Lines with Hough",result);
		imwrite("hough.bmp", result);
	}
   // Create LineFinder instance
	LineFinder ld;

   // Set probabilistic Hough parameters
	ld.setLineLengthAndGap(60,10);
	ld.setMinVote(4);

   // Detect lines
	std::vector<Vec4i> li= ld.findLines(contours);
	Mat houghP(image.size(),CV_8U,Scalar(0));
	ld.drawDetectedLines(houghP);

	if(showSteps){
		namedWindow("Detected Lines with HoughP");
		imshow("Detected Lines with HoughP", houghP);
		imwrite("houghP.bmp", houghP);
	}

   // bitwise AND of the two hough images
	bitwise_and(houghP,hough,houghP);
	Mat houghPinv(image.size(),CV_8U,Scalar(0));
	Mat dst(image.size(),CV_8U,Scalar(0));
	threshold(houghP,houghPinv,150,255,THRESH_BINARY_INV); // threshold and invert to black lines

	image.copyTo(dst, houghPinv); // copy lines to image
	
	std::stringstream stream;
	stream << "Lines Segments: " << lines.size();
	
	putText(dst, stream.str(), Point(10,image.rows-10), 2, 0.8, Scalar(0,0,255),0);
        imshow(window_name, dst); 
	imwrite("processed.bmp", dst);
	char key = (char) waitKey(2);
	lines.clear();
	}
}




