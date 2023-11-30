import cv2
import dlib

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load the pre-trained face landmark predictor from dlib
predictor = dlib.shape_predictor("C:/Users/vanuj/OneDrive/Desktop/genderdetect/shape_predictor_68_face_landmarks.dat")

# Load the pre-trained gender classification model
gender_classifier = cv2.dnn.readNetFromCaffe("C:/Users/vanuj/OneDrive/Desktop/genderdetect/gender_deploy.prototxt", 
                                              "C:/Users/vanuj/OneDrive/Desktop/genderdetect/gender_net.caffemodel")

# Define the gender labels
gender_labels = ['Male', 'Female']

# Define a function to detect gender
def detect_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_classifier.setInput(blob)
    gender_preds = gender_classifier.forward()
    gender = gender_labels[gender_preds[0].argmax()]
    return gender

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Iterate through the detected faces
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the face landmarks
        landmarks = predictor(gray, face)

        # Extract the region of interest (ROI) for gender classification
        face_roi = frame[y:y+h, x:x+w]

        # Detect gender
        gender = detect_gender(face_roi)

        # Display the gender label on the frame
        cv2.putText(frame, f'Gender: {gender}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gender Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
