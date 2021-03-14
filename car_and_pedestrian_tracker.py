import cv2

# importing video
video = cv2.VideoCapture('videoplayback-ped.mp4')

# pre trained car classifier
car_classifier_file = 'car_detector.xml'

# pre trained pedestrian classifier
pedestrian_classifier_file = 'pedestrian_tracker.xml'

# create car classifier
# classifiying what is a car and what not
car_tracker = cv2.CascadeClassifier(car_classifier_file)

# create pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

# looping through each frame of the video to get the car image
while True:
    # reading current frame
    # returns if the read was successful or not and current frame in a tuple
    (read_successful, frame) = video.read()

    # Checking if frame was returned or not
    if read_successful:
        # must convert to gray scale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        # breaks at the last frame of the video
        break

    # detecting cars in the frame
    # multiscale means detect cars of any size
    # it returns list of lists that have coordinates of rectangle that have cars in the image
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    # detecting pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangle around the cars in each frames
    # x,y,w(width) and h(height) are coordinate of the car that comes from multiscale
    # (0,0,255) is color of rectangle and 2 is thickness of rectangle
    # for loop destructures every sublist and gets the x,y,w,h
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # drawing rectangle around pedestrains
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display the image with cars spotted
    # 'Car Detector' will be the title of the image
    cv2.imshow('Car And Pedestrian Detector', frame)

    # stay on the frame for 1 sec
    key = cv2.waitKey(1)

    # stop if Q is pressed
    # key is what we press on keyboard as waitkey
    if key == 81 or key == 113:
        break

# Releasing the video capture object after video is over to release space
video.release()


"""
#code for processing image

#importing image
img_field = 'car_image.jpg'

# creating opencv image
# reads all the pixels in the image and stores the pixels in a multidimensional array
img = cv2.imread(img_field)

# convert to grayscale(needed for HAAR cascade)
# converting color image to black and white image
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
# classifiying what is a car and what not
car_tracker = cv2.CascadeClassifier(classifier_file)

# detecting cars in the image
# multiscale means detect cars of any size
# it returns list of lists that have coordinates of rectangle that have cars in the image
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangle around the cars
# x,y,w(width) and h(height) are coordinate of the car that comes from multiscale
# (0,0,255) is color of rectangle and 2 is thickness of rectangle
# for loop destructures every sublist and gets the x,y,w,h
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with cars spotted
# 'Car Detector' will be the title of the image
cv2.imshow('Car Detector', img)

# Dont auto close(Wait for a key press)
cv2.waitKey()
"""

print("Completed")
