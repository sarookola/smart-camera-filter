import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


bg_image = cv2.imread("1404.jpg")
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB and process with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # Create blurred background version of the frame
        blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Expand the face box a bit
                x1, y1 = max(x - 0, 0), max(y - 0, 0)
                x2, y2 = min(x + w + 0, iw), min(y + h + 0, ih)

                # Draw white box on mask where face is
                mask[y1:y2, x1:x2] = 255

        # Blend background and face
        mask_3d = cv2.merge([mask, mask, mask])
        output = np.where(mask_3d == 255, frame, bg_image)

        cv2.imshow("Shayan Smart Camera Filter", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
