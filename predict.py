import numpy as np
import cv2
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='modelv0.2b.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def extract_frames(video_path, num_frames=10, resize=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(frame_count // num_frames, 1)
    
    for i in range(0, frame_count, step):
        if len(frames) < num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, resize)
                frames.append(frame)
            else:
                break
    
    cap.release()
    return np.array(frames)

def predictor(path):
    frames = extract_frames(path)
    frames = frames.astype('float32') / 255.0
    frames = frames.reshape((-1, 128, 128, 3))

    predictions = []
    for frame in frames:
        interpreter.set_tensor(input_details[0]['index'], [frame])

        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])

    predictions = np.array(predictions)
    #print('Predictions:', predictions)

    average_prediction = np.mean(predictions, axis=0)
    video_class = np.argmax(average_prediction)
    class_names = ['Real', 'Fake']
    threshold = 0.5000000596046448
    if average_prediction[1] > threshold:
        video_class = 1
    else:
        video_class = 0
    return [average_prediction, class_names[video_class]]
    # print(f'Average Prediction: {average_prediction}')
    # print(f'Predicted Video Class: {class_names[video_class]}')
