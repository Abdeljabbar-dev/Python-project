# python Projet_SIC.py --shape-predictor shape_predictor_68_face_landmarks.dat --image img1.jpg
# python Projet_SIC.py --shape-predictor shape_predictor_68_face_landmarks.dat --image img2.jpg
# python Projet_SIC.py --shape-predictor shape_predictor_68_face_landmarks.dat --image img3.jpg
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import numpy
from os.path import join



start_time = time.time()
AP = argparse.ArgumentParser()
AP.add_argument("-p", "--shape-predictor", required=True,
               help="path to facial landmark predictor")
AP.add_argument("-i", "--image", required=True,help="path to input image")
ARGS = vars(AP.parse_args())

#print "telechargement facial landmark predictor..."

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(ARGS["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(LSTART, LEND) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(RSTART, REND) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


PADDING_X = 3
PADDING_Y = 3
EYE_AR_THRESH = 0.3
refObj = None

while True:
    IMAGE = cv2.imread(ARGS["image"])
    IMAGE = imutils.resize(IMAGE,width=1000)
 
    IMAGE = IMAGE[1] if ARGS.get("video", False) else IMAGE

    if IMAGE is None:
        break

    BLURRED = cv2.GaussianBlur(IMAGE, (11, 11), 0)
    GRAY = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)

    # detect faces in the gray scale frame
    FACES = DETECTOR(GRAY, 0)

    for face in FACES:
        facial_landmarks = PREDICTOR(GRAY, face)
        facial_landmarks = face_utils.shape_to_np(facial_landmarks)

        leftEye = facial_landmarks[LSTART:LEND]
        rightEye = facial_landmarks[RSTART:REND]

        # Location de contour  gouche 
        xLeft, yLeft = leftEye[0][0], leftEye[2][1]
        widthL, heightL = leftEye[3][0], leftEye[4][1]
	O=xLeft
	D=widthL
	M=(D-O)/2
	
        # Location de contour  droite 
        xRight, yRight = rightEye[0][0], rightEye[2][1]
        widthR, heightR = rightEye[3][0], rightEye[4][1]
	
	P=xRight
	E=widthR
	N=(E-P)/2
		

	leftEyeHull = cv2.convexHull(leftEye)
	rightEyeHull = cv2.convexHull(rightEye)

        #dessin de contour 
		
	cv2.drawContours(IMAGE, [leftEyeHull], -1, (0, 255,0), 1)
	cv2.drawContours(IMAGE, [rightEyeHull], -1, (0, 255, 0), 1)

        # Extracting region of left eye
        leftPart = GRAY[yLeft + PADDING_Y:heightL, xLeft + PADDING_X:widthL - PADDING_X]
	
        leftPartColor = IMAGE[yLeft + PADDING_Y:heightL, xLeft + PADDING_X:widthL - PADDING_X]
	
	F=xLeft + PADDING_X
	B=widthL - PADDING_X
	
        # Extracting region of right eye
        rightPart = GRAY[yRight + PADDING_Y:heightR - PADDING_Y, xRight + PADDING_X:widthR - PADDING_X]
        rightPartColor = IMAGE[yRight + PADDING_Y:heightR - PADDING_Y, xRight + PADDING_X:widthR - PADDING_X]
             
        # trouver le pupille
        (_, _, minLocL,_) = cv2.minMaxLoc(leftPart)
        cv2.circle(leftPartColor, minLocL,2, (0, 0, 255),2)
	#print "centre de pupil"
	G=minLocL[0]
       
        (_, _, minLocR,_) = cv2.minMaxLoc(rightPart)
        cv2.circle(rightPartColor, minLocR, 2, (0, 0, 255), 2)
	H=minLocL[0]

        if G > M-4 and G < M+4:
		print "milieu"
        if G > M+4:
		print "gauche"
        if G < M-4:
		print "droite"

	if H > N-4 and H < N+4:
		print "milieu"
        if H > N+4:
		print "gauche"
        if H < N-4:
		print "droite"
        print("Temps d execution : %s secondes ---" % (time.time() - 		start_time))
    cv2.imshow("IMAGE", IMAGE)

    KEY = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if KEY == ord("q"):
        break

# clean
cv2.destroyAllWindows()
