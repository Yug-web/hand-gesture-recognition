import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import Image

# Define constants
offset = 20
imgSize = 300
labels = ["Hello", "I love you", "Yes","Thank you","Okay", "Please" ]

# Initialize Hand Detector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

# Streamlit UI
st.title("Real-Time Hand Gesture Recognition")
st.write("Show a hand gesture to the webcam, and the model will predict its label.")
start = st.button("Start Webcam")

# Streamlit container for the video feed
video_feed = st.empty()

if start:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to open the webcam. Please check your device settings.", icon="🚨")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.error("Failed to capture video. Please check your webcam.", icon="🚨")
            break
        
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size != 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Get prediction from classifier
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Draw prediction on output image
                cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                              (x - offset + 200, y - offset), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Convert BGR to RGB for Streamlit
        imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        imgRGB = Image.fromarray(imgRGB)

        # Display the video feed in Streamlit
        video_feed.image(imgRGB)

    cap.release()
    cv2.destroyAllWindows()
