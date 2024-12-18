# Save the picture with name

(1) $ rosrun new_face_recognition take_picture.py -n {name}
(2) $ rosrun new_face_recognition take_ID.py -n {name} -image {address of image}


# Upload the picture to Facebank

$ rosrun new_face_recognition facebank.py


# Execution

$ roslaunch usb_cam usb_cam-test.launch

(1) $ rosrun new_face_recognition face_three.py [-fr] [-er] [-ge]
(2) $ rosrun new_face_recognition face_recognition.py (only face recognition)
(3) $ rosrun new_face_recognition light_face_recognition.py (no image output)

$ rostopic pub --once /face_recognition_msg std_msgs/String "data: 'On'" (or option '-f' in rosrun)
