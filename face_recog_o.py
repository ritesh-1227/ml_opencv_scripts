import face_recognition as fr
import cv2
import numpy as np


# Reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# sample pictures
pic_1 = fr.load_image_file("picture1")
pic1_encoding = fr.face_encodings(pic_1)[0]

pic_2 = fr.load_image_file("picture2")
pic2_encoding = fr.face_encodings(pic_2)[0]

# known face encodings and their names
known_face_encodings = [
    picture1,
    picture2
]
known_face_names = [
    "picture_1",
    "picture_2"
]

face_locations = []
face_encodings = []
face_names = []
frame_pr = True

while True:
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if frame_pr:
        # Find all the faces and face encodings in the current frame of video
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    frame_pr = not frame_pr


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (180, 50, 250), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Streaming Now.....|\', frame)

    # Press spacebar to quit!
    if cv2.waitKey(5) & 0xFF == ord(chr(32)):
        break

# Release the webcam handle
video_capture.release()
cv2.destroyAllWindows()
