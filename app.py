from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import pyttsx3


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
voiceRate = 160
engine.setProperty('rate', voiceRate)


def speak(text):
    engine.say(text)
    engine.runAndWait()


def detect_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (Start_x, Start_y, End_x, End_y) = box.astype('int')

            (Start_x, Start_y) = (max(0, Start_x), max(0, Start_y))
            (End_x, End_y) = (min(w - 1, End_x), min(h - 1, End_y))

            face = frame[Start_y:End_y, Start_x:End_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((Start_x, Start_y, End_x, End_y))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

        return locs, preds


prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model('MaskDetector.model')

speak("This program made for detect mask, whether a person wear a mask or not.")

speak("Your system is ready for start video stream.")
print('[INFO] starting video stream ....')
vs = VideoStream(scr=0).start()

while True:
    try:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        (locs, preds) = detect_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.putText(frame, label, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    except:
        pass

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
