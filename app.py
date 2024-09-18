import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from yolov5 import detect
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device


YOLOV5_PATH = "D:/OpenAIApps/Lauretta/yolov5"  

sys.path.insert(0, str(Path(YOLOV5_PATH)))

def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    """Plots one bounding box on the image."""
    if color is None:
        color = (0, 255, 0)  
    if line_thickness is None:
        line_thickness = 3  

    xyxy = [int(x) for x in xyxy]  # Convert to integers
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=line_thickness)

    if label:
        label_size, _ = cv2.getTextSize(label, 0, 0.05, 1)
        label_ymin = max(xyxy[1], label_size[1] + 10)
        cv2.putText(img, label, (xyxy[0], label_ymin), 0, 0.5, color, 1)


device = select_device('cpu')  
model = torch.load(os.path.join(YOLOV5_PATH, 'yolov5s.pt'), map_location=device)['model'].float().eval()

def detect_people(frame, conf_thres=0.05, iou_thres=0.6):
    img = cv2.resize(frame, (1280, 1280))  
    img = img[:, :, ::-1].transpose(2, 0, 1)  
    img = np.ascontiguousarray(img)  
    
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  
    
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

    people_count = 0
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                
                # Printing each detection

                print(f"Detection: {xyxy}, Confidence: {conf}, Class: {model.names[int(cls)]}")

                if cls == 0:  
                    people_count += 1
                    plot_one_box(xyxy, frame, label=label, color=[255, 0, 0], line_thickness=2)
    
    return people_count, frame

def main():
    video_path = "Drone Footage of Canberras HISTORIC Crowd.mp4"  
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    start_time, end_time = 14, 32
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame, end_frame = int(start_time * fps), int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_people = 0
    frame_count = 0

    while cap.isOpened() and frame_count < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        
        print(f"Processing frame {frame_count + start_frame}")
        people_count, annotated_frame = detect_people(frame)
        total_people += people_count
        frame_count += 1

        
        cv2.imshow("Detections", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_people = total_people / frame_count if frame_count > 0 else 0
    print(f"Estimated average crowd size: {avg_people:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
