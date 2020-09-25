#USAGE: python fastFaceLandmarkDetection.py

# import the necessay libraries
import cv2,dlib
import sys
import numpy as np


# The below method is called by all time from the faceLandmarkPoints methods inside and then we pass here image and points,
# on the behalf of that points we are getting x cordinates and y cordinates of the passing points value and then we draw that points
# on the image face through the cv2.polylines.
def drawPointsFace(image, faceLandmarks, start, end, isClosed=False):
  facePoints = []
  for i in range(start, end+1):
    point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
    facePoints.append(point)

  points = np.array(facePoints, dtype=np.int32)
  cv2.polylines(image, [points], isClosed, (150, 150, 0), thickness=2, lineType=cv2.LINE_8)

# the below method will first check either points exactly 68 or not. If assertion becomes true then it will call
# one by one drawPointsFace method to draw on the image which is actually a frame which coming through the web camera.
def faceLandmarkPoints(image, faceLandmarks):
    assert(faceLandmarks.num_parts == 68)
    drawPointsFace(image, faceLandmarks, 0, 16)           # Jaw line
    drawPointsFace(image, faceLandmarks, 17, 21)          # Left eyebrow
    drawPointsFace(image, faceLandmarks, 22, 26)          # Right eyebrow
    drawPointsFace(image, faceLandmarks, 27, 30)          # Nose bridge
    drawPointsFace(image, faceLandmarks, 30, 35, True)    # Lower nose
    drawPointsFace(image, faceLandmarks, 36, 41, True)    # Left eye
    drawPointsFace(image, faceLandmarks, 42, 47, True)    # Right Eye
    drawPointsFace(image, faceLandmarks, 48, 59, True)    # Outer lip
    drawPointsFace(image, faceLandmarks, 60, 67, True)    # Inner lip


# Here we are loading the Dlib 68 face landmark model
modelPath = "shape_predictor_68_face_landmarks.dat"

# Line 13, to process the fast detection, as we told before in the above theory we have to fixed the size of the frame and run the
# face detection and landmarks on that frame and later we scale the output co-ordinates value with the original frame. So
# here we kept the size of the frame is 480.
heightResize = 480

# Line 17, here we are specifying how many frames it has to skipped during live so that it will deduct fast face detection. 
framesSkipping = 2

try:
  # Line no. 21, we are creating a window name
  windowName = "Detecting the Facial Landmark Fast"

  # Line 22, here we are creating a VideocameraObjectture object.
  cameraObject = cv2.VideoCapture(0)

  # Line 29, here the video object is trying to find out either it can read the frame or not which means,
  # sometimes the webcam is not working properly or webcam off, so this fucntion will care of it, so that without the frame read programme will go
  # unconditionally stop. To overcome of that, we use this method to inform user, kindly check your webcam.
  if (cameraObject.isOpened() is False):
    print("Unable to connect to camera, kindly check your web camera.")
    sys.exit()

  # Line 34, here we just keep a value but this is not original value. The Actual value calculated after 100 frames.
  framePerSecond = 30.0

  # Line 37, it will read the first frame using video object.
  ret, image = cameraObject.read()


  # Line 42 - 49, these lines first check the height of the coming frame from webcam and then resize height of that frame with the help of the
  # our defined height value. If this is not happend, then it's means frames not able to read and it will simply exit. 
  if ret == True:
    height = image.shape[0]
    # calculate resize scale
    frame_resize_scale = float(height)/heightResize
    size = image.shape[0:2]
  else:
    print("Unable to read frame")
    sys.exit()


  # Line 53 - 54, we loading the face detection and shape predictor models from dlib predefined models
  faceDetector = dlib.get_frontal_face_detector()
  shapePredictor = dlib.shape_predictor(modelPath)
  # Line 56, initiating the tickCounter, which we will used to calculate the actual framePerSecond (frame per second) value.
  time = cv2.getTickCount()

  # Line 60, this count variable we will be use to count each frame, because final value of framePerSecond will calculate on after 100 frames,
  # so this varibale track when 100 frames completed and then update the framePerSecond value.
  count = 0

  # Line 63 - 122 Grab the frame and process the frames until the main window is closed by the user.
  while(True):
    if count==0:
      time = cv2.getTickCount()

    # Line 68, here it's Grab a frame and store this in varibale imageFrame
    ret, image = cameraObject.read()

    # Line 71, we just converting the imageFrame from BGR to RGB format
    imageDlib = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Line 74, now we are creating a frameSmall by resizing image by resize scale 
    imageSmall= cv2.resize(image, None, fx = 1.0/frame_resize_scale, fy = 1.0/frame_resize_scale, interpolation = cv2.INTER_LINEAR)

    # Line 77, we are now converting the frameSmall (image) to again BGR to RGB
    imageSmallDlib = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2RGB)

    # Line 83, this will helps to increase the detection by skipping the frame. The value of skipping frame depends upon 
    # your system hardware and the camera (how much framePerSecond process). This is the main concepts, which can reduce the 
    # computation.

    if (count % framesSkipping == 0):
      # Line 85, detect faces on frameSmall which we resize it already.
      faces = faceDetector(imageSmallDlib,0)

    # Line 88, iterate over faces
    for face in faces:
      # Line 92 -96, as we run face detection on a resized image for faster detection,
      # so, now we will scale up that coordinates (x, y, w, h) value with the original frame, so that we can get face 
      # rectangle on the original frame.
      newRectValues = dlib.rectangle(int(face.left() * frame_resize_scale),
                               int(face.top() * frame_resize_scale),
                               int(face.right() * frame_resize_scale),
                               int(face.bottom() * frame_resize_scale))

      # Line 100, now we are passing two parameters in the predictor () method. One is imDlib which is original frame, as we converted this
      # to BGR to RGB before and second parameters is newRectValues which has (x, y, w, h) rectangle values which detect face on frame. So
      # now to find face landmarks by providing reactangle for each face.
      shape = shapePredictor(imageDlib, newRectValues)
      # Draw facial landmarks
      faceLandmarkPoints(image, shape)

    # Put framePerSecond on the ouput screen at which we are processing camera feed
    cv2.putText(image, "{0:.2f}-framePerSecond".format(framePerSecond), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
    # Display it all on the screen
    cv2.imshow(windowName, image)
    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF

    # Stop the program.
    if key==27:  # ESC
      # If ESC is pressed, exit.
      sys.exit()

    # increment frame counter
    count = count + 1
    # calculate framePerSecond at an interval of 100 frames
    if (count == 100):
      time = (cv2.getTickCount() - time)/cv2.getTickFrequency()
      framePerSecond = 100.0/time
      count = 0

  #cv2.destroyAllindows() will destroy all windows which we created till now. If you want to destroy any particular window then
  # we have to use the cv2.destroyWindow() and pass the exact window name as argument inside of this function.
  cv2.destroyAllWindows()

  # this is basically to release the device which was used by the program, if it not release then other device not able to use that device and it will
  # raise errors.
  cameraObject.release()

except Exception as e:
  print(e)
