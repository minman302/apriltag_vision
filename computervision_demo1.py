# This module runs on a Raspberry Pi with wpilibpi image.
# Hardware: Raspberry Pi 3B, Logitech C920 WebCam.
import time
import os
import json
import numpy as np

import cv2
import robotpy_apriltag
# import ntcore
from wpimath.geometry import Transform3d


DEBUG = False
DEBUG1 = False
USE_TEST_FILES = False
CAPTURE_IMAGES = False

# constructor for Parameter class that holds camera and AprilTag parameters (resolution, fps... )
# camera parameters are stored externally in a JSON file
class Parameters:
    def __init__(self,json_file='parameters.json'):
        with open("parameters.json", "r") as read_file:
            self.parameters = json.load(read_file)

    def get_camera_param(self):
        return self.parameters['Camera']

    def get_post_process(self):
        return self.parameters['post_process']

    def get_apriltag_para(self):
        return self.parameters['apriltag_detection']


if __name__ == '__main__':
    para = Parameters()
    camera_para = para.get_camera_param()
    post_process_para = para.get_post_process()
    apriltag_para = para.get_apriltag_para()

    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_para['frame_width'])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_para['frame_height'])
    cam.set(cv2.CAP_PROP_FPS, camera_para['fps'])
    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cam.get(cv2.CAP_PROP_FPS))
    assert actual_width == camera_para['frame_width'], "Frame width does not match setting"
    assert actual_height == camera_para['frame_height'], "Frame height does not match setting"
    assert actual_fps == camera_para['fps'], "Video fps does not match setting"
    discrete_interval = 1.0 / actual_fps

    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily(apriltag_para["tag_family"])
    estimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            apriltag_para["tag_size"],
            camera_para["fx"],
            camera_para["fy"],
            camera_para["cx"],
            camera_para["cy"]
        )
    )

    # Recommended by PhotonVision. Not sure if it is hardware related.
    DETECTION_MARGIN_THRESHOLD = apriltag_para['detection_margin_threshold']  
    DETECTION_ITERATIONS = apriltag_para['detection_iterations']

    cornersBuf = tuple([0.0] * 8)

    img_num = 1
    looping = True
    captured_new_image = False

    last_capture_time = 0
    
    # main camera loop
    while looping:
        if USE_TEST_FILES:
            filename = "original" + str(img_num) + ".png"
            if os.path.isfile(filename):
                input_img = cv2.imread(filename)
                captured_new_image = True
        else:
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

            # Coordinates of found targets, for NT output:
            x_list = []
            y_list = []
            id_list = []

            
            # image processing
            prev_time = time.time()  # start timer for image processing.
            grayimg = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            print("gray image size is: ", grayimg.shape)

            # look for tags
            tag_info = detector.detect(grayimg)
            if not tag_info:
                if DEBUG and DEBUG1:
                    cv2.imwrite("original" + str(img_num) + ".png", input_img)
                print("Did not find AprilTags.")
            else:
                if DEBUG:
                    cv2.imwrite("original" + str(img_num) + ".png", input_img)
                print("AprilTag Detected.")
                # found some tags, report them and update the camera image

                # filter out aprilTags below decision margin to delete "bad" tags
                filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
                for tag in tag_info:
                    # print(dir(tag))
                    print("tag_id: %s, center: %s" % (tag.getId(), tag.getCenter()))
                    print(tag.getCorner(0), tag.getCorner(1), tag.getCorner(2), tag.getCorner(3))
                    print('DecisionMargin')
                    print(tag.getDecisionMargin())
                    print("family:", tag.getFamily())
                    print('Hamming', tag.getHamming())
                    print('Homography', tag.getHomography())
                    print('HomographyMatrix', tag.getHomographyMatrix())
                    print('Corners:')
                    print(tag.getCorners(cornersBuf))
                    # print("cornersBuf",cornersBuf)
                    tag_id = tag.getId()
                    center = tag.getCenter()
                    # hamming = tag.getHamming()
                    # decision_margin = tag.getDecisionMargin()

                    est = estimator.estimateOrthogonalIteration(tag, DETECTION_ITERATIONS)

                    #based off our camera, we can only reduce possible poses of aprilTag to two
                    # however, with human intuition we can get down to one pose [TODO below]
                    pose1 = est.pose1
                    pose2 = est.pose2
                    # TODO: Check if there could be a second pose estimation.
                    # from estimator.estimateOrthogonalIteration(tag, DETECTION_ITERATIONS)
                    print(f"tag ID is {tag_id}, post1 at: {pose1}")
                    print(f"tag ID is {tag_id}, post2 at: {pose2}")

                    detected_center = tag.getCenter()
                    
                    for i in range(4):
                        # Highlight the edges of all recognized tags and label them with their IDs:
                        if ((tag_id > 0) & (tag_id < 9)):
                            col_box = (0, 255, 0)
                            col_txt = (255, 255, 255)
                        else:
                            col_box = (0, 0, 255)
                            col_txt = (0, 255, 255)

                        # Draw a frame around the tag:
                        corner0 = (int(tag.getCorner(0).x), int(tag.getCorner(0).y))
                        corner1 = (int(tag.getCorner(1).x), int(tag.getCorner(1).y))
                        corner2 = (int(tag.getCorner(2).x), int(tag.getCorner(2).y))
                        corner3 = (int(tag.getCorner(3).x), int(tag.getCorner(3).y))
                        cv2.line(output_img, corner0, corner1, color=col_box, thickness=2)
                        cv2.line(output_img, corner1, corner2, color=col_box, thickness=2)
                        cv2.line(output_img, corner2, corner3, color=col_box, thickness=2)
                        cv2.line(output_img, corner3, corner0, color=col_box, thickness=2)

                        # Label the tag with the ID:
                        cv2.putText(output_img, f"{tag_id}", (int(center.x), int(center.y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    col_txt,
                                    2)

                        x_list.append((center.x - actual_width / 2) / (actual_width / 2))
                        y_list.append((center.y - actual_width / 2) / (actual_width / 2))
                        id_list.append(tag_id)

                start_time = time.time()
                processing_time = start_time - prev_time
                prev_time = start_time

                fps = 1 / processing_time
                print(f"Image processing rate in {processing_time}s, or {fps} fps")
                cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

                #for every 10th image,
                if img_num % 10 == 0:
                    if DEBUG:
                        cv2.imwrite("processed" + str(img_num) + ".png", input_img)
                        cv2.imwrite("final" + str(img_num) + ".png", output_img)
                # stop after 500 images (5.55s)
                if img_num > 500:
                    looping = False
            img_num += 1

            # refresh the camera image
            # cv2.imshow('Result', image)
            # let the system event loop do its thing
            key = cv2.waitKey(20)
            # terminate the loop if the 'Return' key his hit
            if key == 13:
                looping = False
