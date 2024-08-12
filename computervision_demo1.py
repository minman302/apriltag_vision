# This module runs on a Raspberry Pi with wpilibpi image.
# Hardware: Raspberry Pi 3B, Logitech C920 WebCam.
import time
import os
import json
import numpy as np

import cv2
import robotpy_apriltag
# import ntcore
import wpimath.geometry


DEBUG = False
DEBUG1 = False
USE_TEST_FILES = False
CAPTURE_IMAGES = False


#pose object that holds our robot's position
robotPose = wpimath.geometry.Pose2d();

# aprilTag positions: NWU coordinate system in the bottom left
aprilTagList = []
for i in range(1, 17):
    #Create a temporary aprilTag object to modify below
    tempTag = wpimath.geometry.Pose2d();

    #Modify each aprilTag object based on which wall it is on
    if i < 5:
        tempTag.rotateBy(wpimath.geometry.Rotation2d(np.pi * 3.0 / 4.0))
        tempTag.transformBy(wpimath.geometry.Transform2d(wpimath.geometry.Translation2d(i, 0), wpimath.geometry.Rotation2d()))
    elif i < 9:
        tempTag.rotateBy(wpimath.geometry.Rotation2d(np.pi / 2.0))
        tempTag.transformBy(wpimath.geometry.Transform2d(wpimath.geometry.Translation2d(5, 4 - i), wpimath.geometry.Rotation2d()))
    elif i < 12:
        tempTag.rotateBy(wpimath.geometry.Rotation2d(np.pi / 4.0))
        tempTag.transformBy(wpimath.geometry.Transform2d(wpimath.geometry.Translation2d(13 - i, 5), wpimath.geometry.Rotation2d()))
    else:
        tempTag.rotateBy(wpimath.geometry.Rotation2d())
        tempTag.transformBy(wpimath.geometry.Transform2d(wpimath.geometry.Translation2d(0, 17 -i), wpimath.geometry.Rotation2d()))

    #append the modified aprilTag to the list. The list index matches the aprilTag ID number
    aprilTagList.append(tempTag)


if __name__ == '__main__':

    cam = cv2.VideoCapture(0)

    frame_width = 1920
    frame_height = 1200
    fps = 5

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cam.set(cv2.CAP_PROP_FPS, fps)
    
    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cam.get(cv2.CAP_PROP_FPS))

    assert actual_width == frame_width, "Frame width does not match setting"
    assert actual_height == frame_height, "Frame height does not match setting"
    assert actual_fps == fps, "Video fps does not match setting"
    
    discrete_interval = 1.0 / actual_fps

    tag_family = "tag36h11" 
    tag_size = 0.1651 # 6.5in -> .1651m
    
    # modify these values after discussion with Ryan
    # Calibrated values
    fx = 595.7
    fy = 595.1
    cx = 948.4 
    cy = 507.1 

    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily(tag_family)
    estimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            tag_size,
            fx,
            fy,
            cx,
            cy
        )
    )

    # Recommended by PhotonVision. Not sure if it is hardware related.
    DETECTION_MARGIN_THRESHOLD = 30 # set this value, just placeholder rn
    DETECTION_ITERATIONS = 50 # set this value, just placeholder rn
    
    cornersBuf = tuple([0.0] * 8)

    img_num = 1
    looping = True
    captured_new_image = False

    last_capture_time = 0
    
    # main camera loop
    while looping:

        capture_time = time.time()

        #if the capture is a new frame
        if capture_time - last_capture_time > discrete_interval:
            result, input_img = cam.read() # read the frame
            last_capture_time = capture_time #reset the last capture time
            captured_new_image = True
        else:
            # wait a tiny amount of time and check if there is a new frame again
            time.sleep(0.001)

        #if we have a new frame
        if captured_new_image:
            
            # we no longer are looking for a new frame
            captured_new_image = False
            output_img = np.copy(input_img)

            
            # image processing
            prev_time = time.time()  # start timer for image processing.
            grayimg = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

            # look for tags
            tag_info = detector.detect(grayimg)
            if not tag_info:

                # filter out aprilTags below decision margin to delete "bad" tags
                filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]

                for tag in tag_info:
                    tag_id = tag.getId()
                    est = estimator.estimateOrthogonalIteration(tag, DETECTION_ITERATIONS)

                    #based off our camera, we can only reduce possible poses of aprilTag to two
                    # however, with human intuition we can get down to one pose [TODO below]
                    pose1 = est.pose1
                    pose2 = est.pose2

                    # TODO: Check if there could be a second pose estimation.
                    # For now, use pose with lowest objest space error:
                    if est.error1 < est.error2:
                        rightPose = pose1
                    else:
                        rightPose = pose2

                    print("Tag ID: %s, Tag Pose: X: %3d, Y: %3d, Angle: %3d" % (tag_id, rightPose.X, rightPose.Y, rightPose.rotation.toRotation2d.X))

            

                    # robot pose estimation
                    

                start_time = time.time()
                processing_time = start_time - prev_time
                prev_time = start_time

                fps = 1 / processing_time
                print(f"Image processing rate in {processing_time}s, or {fps} fps")
                cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

                # #for every 10th image,
                # if img_num % 10 == 0:
                #     if DEBUG:
                #         cv2.imwrite("processed" + str(img_num) + ".png", input_img)
                #         cv2.imwrite("final" + str(img_num) + ".png", output_img)
                # # stop after 500 images (5.55s)
                # if img_num > 500:
                #     looping = False
            img_num += 1

            # refresh the camera image
            # cv2.imshow('Result', image)
            # let the system event loop do its thing
            key = cv2.waitKey(20)
            # terminate the loop if the 'Return' key his hit
            if key == 13:
                looping = False
