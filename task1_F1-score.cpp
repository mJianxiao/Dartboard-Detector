/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread( "dart14.jpg", CV_LOAD_IMAGE_COLOR);

       	// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );


        rectangle(frame,Point(460,210),Point(560,310),Scalar(0,0,255),2);
        rectangle(frame,Point(720,190),Point(820,290),Scalar(0,0,255),2);

        

	// 4. Save Result Image
        imwrite( "f1-score.jpg", frame );
	return 0;
}



/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

        int truth = 2;
        int size = 100;
        int m=0,n=0;
        for(int i = 0; i < faces.size();i++)
	  {	 
	    if (-10<(faces[i].height - size) && (faces[i].width - size)<10 && -10<(faces[i].height - size) && (faces[i].width - size) <10 && ((-10<((faces[i].x +faces[i].width/2) - 510) && ((faces[i].x+faces[i].width/2) - 510) < 10 && -10<((faces[i].y+faces[i].height/2) - 260)  && ((faces[i].y +faces[i].height/2) - 260< 10)) || (-10< ((faces[i].x +faces[i].width/2) - 770) && ((faces[i].x+faces[i].width/2) - 770)<10 && -10<((faces[i].y+faces[i].height/2) - 240) && ((faces[i].y+faces[i].height/2) -240)<10)))
	    { m++;}
	  else
            {n++;}
       }
        int p = truth - m;
	  if ((m+n == 0) || (m+p == 0))
          {printf("fail detect\n");}
        double precision = m/(m+n);
        double recall = m/(m+p);
        double f = 2*precision*recall/(precision+recall);
        std::cout << f << std::endl;


}

