import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
import torchvision.transforms as T


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()


transform = T.Compose([
    T.ToTensor()
])

def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    """Plots one bounding box on the image."""
    if color is None:
        color = (0, 255, 0)  
    if line_thickness is None:
        line_thickness = 3  

    xyxy = [int(x) for x in xyxy]  
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=line_thickness)

    if label:
        label_size, _ = cv2.getTextSize(label, 0, 0.05, 1)
        label_ymin = max(xyxy[1], label_size[1] + 10)
        cv2.putText(img, label, (xyxy[0], label_ymin), 0, 0.5, color, 1)

def process_frame(frame):
    """Process a single video frame for object detection."""
    image_tensor = transform(frame).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    
    
    for i in range(len(boxes)):
        if scores[i] > 0.5:  
            box = boxes[i].astype(int)
            label = labels[i]
            label_name = 'person' if label == 1 else 'unknown'
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{label_name} {scores[i]:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

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

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > end_frame:
            break
        
        
        frame = process_frame(frame)
        
        
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
        total_people += sum(1 for i in range(len(frame['boxes'])) if frame['scores'][i] > 0.5 and frame['labels'][i] == 1)
        frame_count += 1
    
    avg_people = total_people / frame_count if frame_count > 0 else 0
    print(f"Estimated average crowd size: {avg_people:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
