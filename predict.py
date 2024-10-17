import numpy as np
import cv2
import tensorflow as tf

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

interpreter = tf.lite.Interpreter(model_path='modelv0.2b.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def extract_frames(video_path, num_frames=10, resize=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    face_detected = False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(frame_count // num_frames, 1)
    
    for i in range(0, frame_count, step):
        if len(frames) < num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    face_detected = True  

                frame = cv2.resize(frame, resize)
                frames.append(frame)
            else:
                break
    
    cap.release()
    return np.array(frames), face_detected

def predictor(path):
    frames, face_detected = extract_frames(path)
    
    if not face_detected:
        return [0, "No Face Detected"]

    frames = frames.astype('float32') / 255.0
    frames = frames.reshape((-1, 128, 128, 3))

    predictions = []
    for frame in frames:
        interpreter.set_tensor(input_details[0]['index'], [frame])

        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])

    predictions = np.array(predictions)
    

    average_prediction = np.mean(predictions, axis=0)
    class_names = ['Real', 'Fake']
    
    threshold = 0.5000000596046448
    if average_prediction[1] > threshold:
        video_class = 1
    else:
        video_class = 0
        
    return [average_prediction[1], class_names[video_class]]
