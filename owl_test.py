import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "a cat. a remote control."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)


vid = cv2.VideoCapture(0)   # 0 for logitech; 4 for realsense
while True:
                ret, frame = vid.read()
                cv2.imshow('Image Capture', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or img_counter == self.image_threshold:
                    break
                
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results, xmin, ymin, xmax, ymax = self.owl_pred_live(frame_bgr, box_threshold, text_threshold)