import os
import torch
import cv2
import json
import yaml
import splitfolders
import numpy as np
from PIL import Image
from datetime import datetime
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO
import pyrealsense2 as rs
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class NewData:
    def __init__(self, combined_folder, json_file, object, image_threshold, epochs, map_threshold, inference, inference_threshold):
        self.combined_folder = combined_folder
        self.json_file = json_file
        self.object = object
        self.image_threshold = image_threshold
        self.epochs = epochs
        self.map_threshold = map_threshold
        self.inference = inference
        self.inference_threshold = inference_threshold

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.device = torch.device(0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        self.model_yolov8 = YOLO('yolov8n.pt')

    
    def owl_pred_live(self, color_frame, box_threshold=0.6, text_threshold=0.4):
        xmin = ymin = xmax = ymax = None
        image = Image.fromarray(color_frame)  # for live feed
       
        inputs = self.processor(text=self.object, images=image, return_tensors="pt").to(self.device)
        torch.cuda.empty_cache()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
        # target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]])
        result = results[0]

        if len(result['labels']) != 0:  # Check if labels list is not empty
            xyxy = result["boxes"][0]
            xmin, ymin, xmax, ymax = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()

        return results, xmin, ymin, xmax, ymax




    #testing
    def capture_pred(self, box_threshold, text_threshold):
        # Capture using cv2
        vid = cv2.VideoCapture(0)   # 0 for logitech; 4 for realsense
        try:
            img_counter=0
            while True:
                ret, frame = vid.read()
                cv2.imshow('Image Capture', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or img_counter == self.image_threshold:
                    break
                
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results, xmin, ymin, xmax, ymax = self.owl_pred_live(frame_bgr, box_threshold, text_threshold)
                # Store only if object is detected in frame
                if xmin != None:
                    img_folder = self.combined_folder+"/raw_dataset/images"
                    txt_folder = self.combined_folder+"/raw_dataset/labels"
                    if not os.path.exists(img_folder):
                        os.makedirs(img_folder)
                        os.makedirs(txt_folder)
                    img_name = f"image_{img_counter}_{self.timestamp}.jpg"
                    img_path = os.path.join(img_folder, img_name)
                    ih, iw = frame.shape[:2]
                    image_sh = cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)
                    cv2.imshow("Detected OWL", image_sh)
                    cv2.imwrite(img_path, image_sh)
                    print(f"{img_name} written!")
                    img_counter += 1
                    # Create labels.txt
                    txt_path = os.path.join(txt_folder, os.path.splitext(img_name)[0] + ".txt")
                    with open(self.json_file, 'r') as file:
                        data = json.load(file)
                    label_number = len(data['candidate_labels'])-1
                    with open(txt_path, 'w') as file:
                        xc = (xmin + xmax)/2
                        yc = (ymin + ymax)/2
                        w = xmax - xmin
                        h = ymax - ymin
                        data = f"{label_number} {xc/iw} {yc/ih} {w/iw} {h/ih}"
                        file.write(data)
            
                if img_counter == self.image_threshold:
                    break
                    
        finally:
            vid.release()
            cv2.destroyAllWindows()

    def split_and_yaml(self):
        upper_folder = self.combined_folder+"/split_dataset"
        if not os.path.exists(upper_folder):
            os.makedirs(upper_folder)
        splitfolders.ratio(self.combined_folder+"/aug_dataset", output=upper_folder, ratio=(0.7, 0.3))
        print("Training and validation sets ready")
        # Create yaml file
        f = open(self.json_file)
        candidate_labels = json.load(f)['candidate_labels']
        f.close()
        path = os.path.abspath(self.combined_folder)
        split_path = path.split("/{}".format(self.combined_folder))[0]
        yaml_content={
            'path': split_path,
            'train': f"{upper_folder}/train",
            'val': f"{upper_folder}/val",
            'names': {index: value for index,value in enumerate(candidate_labels)}
        }
        yaml_path = self.combined_folder+"/train.yaml"
        if not os.path.exists(yaml_path):
            with open(yaml_path,'w') as file:
                yaml.dump(yaml_content, file)
        print("YAML file created")

    def train(self):
        results = self.model_yolov8.train(data=f"{self.combined_folder}/train.yaml", epochs=self.epochs, device=self.device, project=self.combined_folder)
        rdict = results.__dict__
        new_weights_path = str(rdict["save_dir"])+"/weights/best.pt"
        # Get MaP50 Score
        map = rdict['box'].__dict__
        map50 = map['all_ap'][0][0]
        # Put threshold on MaP50 score
        if map50>=self.map_threshold:
            new_weights_path = new_weights_path
            print("Trained and stored the new weights")
        else:
            new_weights_path = None
            print('Try with more images and training more epochs')
        # Start live inference
        if self.inference and new_weights_path!=None:
            vid = cv2.VideoCapture(0)
            new_yolov8 = YOLO(new_weights_path).to(self.device)
            i = 0
            while True:
                _, frame = vid.read()
                cv2.imshow('Image Capture', frame)
                predictions = new_yolov8(frame, conf=self.inference_threshold)
                for p in predictions:
                    bclmt = p.boxes.conf
                    if bclmt.nelement()!=0:
                        print("Detected!")
                        upper_folder_results = f"{self.combined_folder}/pred_results"
                        if not os.path.exists(upper_folder_results):
                            os.makedirs(upper_folder_results)
                        img_bgr = p.plot()
                        img_rgb = Image.fromarray(img_bgr[..., ::-1])
                        img_rgb.show()
                        img_rgb.save(f"{upper_folder_results}/result{i}.jpg")
                        i += 1
                else:
                    print("No detection")

                if cv2.waitKey(1) & 0xFF==ord('q'):
                    break
        else:
            predictions = None

        return new_weights_path, map50, predictions
