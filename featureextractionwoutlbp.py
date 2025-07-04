import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern

dataset = "cleaned_dataset"
range_subfol = range(1, 12)

eyecascade_p = cv2.data.haarcascades + 'haarcascade_eye.xml'  #path to the xml file that contains the trained classifier data for eye detection
eye_cascade = cv2.CascadeClassifier(eyecascade_p)

mouthcascade_p = cv2.data.haarcascades + 'haarcascade_smile.xml' #path to mouth xml file
mouth_cascade = cv2.CascadeClassifier(mouthcascade_p) #creates a mouth classifier object that reads the pre-trained ml model



for subfoldernum in range_subfol:
    subfoldername = f"{subfoldernum:02d}"  
    currfol = os.path.join(dataset, subfoldername) #connects cleaned dataset with the subfolders 01 to 11
    images = [f for f in os.listdir(currfol) if f.endswith(('.jpg'))] #makes a list of all images stored in the subfolder

    output_folder = f"person{subfoldername}_withoutlbp"
    if not os.path.exists(output_folder):
         os.makedirs(output_folder)


    for imagenum in images:
            imagep = os.path.join(currfol, imagenum) #connects subfolders current 
            image = cv2.imread(imagep,0) #loads image into image variable as numpy array

            croppedupper = image[0:125, 0:256]  #crops upper portion of face for accurate detection of eyes
            croppedlower = image[120:256, 0:256] #crops lower portion of face for accurate detection of mouth

            #eyes detection

            eyes = eye_cascade.detectMultiScale(croppedupper, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            #detectmultiscale is a method of cascadeclassifier object eyecascade which does object detection
            #returns a list of rectangles where eyes are detected
            #scale factor shrinks down the image by 10 percent to search for eyes at diff sizes
            #min neighbor draws minimum in this case 5 rectangles around our target so that the ml model considers the eye valid
            #minimum area for detection is minisize checks regions 20x20 in size only

            if len(eyes) > 0:                        
                for (ex, ey, ew, eh) in eyes:  #ex,ey are top left coordinates of the detected eye and ew eh are width and height of the box around detected eye
                   #stores detected exey ewed values in variables so that we can extract the eyes image later 
                   detectedex = ex
                   detectedey = ey
                   detectedeh = eh
                   detectedew = ew

                   edetected = image[detectedey:detectedey+detectedeh, detectedex:detectedex+detectedew]
            
            #mouth detection
            mouths = mouth_cascade.detectMultiScale(croppedlower, scaleFactor=1.1, minNeighbors=15, minSize=(30, 20))
            if len(mouths) > 0:
               for (x, y, w, h) in mouths:  
                
                   detectedmx = x
                   detectedmy = y + 120 #since we cropped from row 120 so we add 120 to the y
                   detectedmh = h
                   detectedmw = w
            
                   mdetected = image[detectedmy:detectedmy+detectedmh, detectedmx:detectedmx+detectedmw]
            else:
                mdetected = image[150:210, 40:200]   #manual hard code estimated extraction in case haar cascade detection fails


            #nose detection (hard coded as its 256x256 so estimated location)
            ndetected = image[70:150, 65:195]

            fdetected = image

            #resizing each image
            edetectedr = cv2.resize(edetected, (128, 128), interpolation=cv2.INTER_AREA)
            mdetectedr = cv2.resize(mdetected, (128, 128), interpolation=cv2.INTER_AREA)
            ndetectedr = cv2.resize(ndetected, (128, 128), interpolation=cv2.INTER_AREA)
            fdetectedr = cv2.resize(fdetected, (128, 128), interpolation=cv2.INTER_AREA)

            # Convert each resized region to a 1D vector
            eye_vector = edetectedr.ravel()
            face_vector = fdetectedr.ravel()
            nose_vector = ndetectedr.ravel()
            mouth_vector = mdetectedr.ravel()

            # Concatenate all vectors into a single array
            concatenated_vector = np.concatenate((eye_vector, face_vector, nose_vector, mouth_vector))

            # Save the concatenated vector as a CSV file
            concatenated_2d = concatenated_vector.reshape(1, -1)
            output_csv_path = os.path.join(output_folder, f"{os.path.splitext(imagenum)[0]}_raw_concat.csv")
            np.savetxt(output_csv_path, concatenated_2d, delimiter=',', fmt='%d')
            print(f"Saved concatenated raw vector as CSV to {output_csv_path}")


