from datetime import datetime
import json
import os
import shutil
import argparse
import logging
from roboflow_bb import RoboflowBB
from new_data import NewData
from utils_aug import Augment
from available_cam import AvailableCam

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoTrain:
    def __init__(self, new_weights, prev_data_folder, roboflow, abs_yaml_file, image_threshold, number_aug, epochs, map_threshold, inference, inference_threshold) -> None:
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        self.new_weights = new_weights
        self.prev_data_folder = prev_data_folder if not new_weights else None
        if not new_weights and prev_data_folder==None:
            logger.error("Correct prev_data_folder must be provided if new_weights is False")
        self.combined_folder = f"data/train_v{self.timestamp}" if not new_weights else f"data/new_weights_v{self.timestamp}"
        self.json_file = f"{self.combined_folder}/inputs.json"
        self.roboflow = roboflow
        if not roboflow and abs_yaml_file is None:
            logger.error("yaml_file must be provided if roboflow is False")
        self.abs_yaml_file = abs_yaml_file if not roboflow else None
        self.image_threshold = image_threshold
        self.number_aug = number_aug
        self.epochs = epochs
        self.map_threshold = map_threshold
        self.inference = inference
        self.inference_threshold = inference_threshold

    def prev_data(self):
        '''
        Function returns previous data in annotated form.
        If the previous data is in roboflow format, convert it to yolov8 format needed
        '''
        rfbb = RoboflowBB(prev_folder=self.prev_data_folder, combined_folder=self.combined_folder, json_file=self.json_file, abs_yaml_file=self.abs_yaml_file)
        if self.roboflow:
            # copies and draws bb in a combined dataset; updates json file as per the yaml file
            rfbb.run()
        else:
            # copies data in combined dataset; and updates json file as per the yaml file
            rfbb.make_copy_folder()
            logger.info("Stored combined data \n")
            rfbb.update_json_from_yaml()
            logger.info("We have updated json file now \n")
    
    def augment(self):
        aug = Augment(combined_folder=self.combined_folder, json_file=self.json_file)
        imgs = [img for img in os.listdir(self.combined_folder+"/raw_dataset/images") if aug.is_image_by_extension(img)]

        for img_file in imgs:
            image, gt_bboxes, aug_file_name = aug.get_inp_data(img_file)
            for n in range(self.number_aug):
                aug_img, aug_label = aug.get_augmented_results(image, gt_bboxes)
                aug.store_aug(aug_img, aug_label, f"{aug_file_name}_{n+1}")
        logger.info("Augmented and saved dataset")

    def new_data(self, object, object_specific):
        '''
        Generates new data for the input object, splits it and creates a YAML file for training
        Trains data to generate new weights file

        Args:
            object = (str) object to be detected
        Returns:
            new_weights_path = (str) Path to the new pt weights file
        '''
        zsl = NewData(combined_folder=self.combined_folder, json_file=self.json_file, object=object, image_threshold=self.image_threshold, epochs=self.epochs, map_threshold=self.map_threshold, inference=self.inference, inference_threshold=self.inference_threshold)
        # Capture, split and store dataset; create yaml file
        zsl.capture_pred(box_threshold=0.6, text_threshold=0.4)
        logger.info("Done capturing frames \n")
        # update the json file with new class
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        data['candidate_labels'].pop()
        data['candidate_labels'].append(object_specific)
        with open(self.json_file, 'w') as file:
            json.dump(data, file, indent=4)
        # Augment dataset
        self.augment()
        # Update yaml file
        zsl.split_and_yaml()
        # Train on new yaml file and get the MaP50 scores
        new_weights_path, _, predictions = zsl.train()
        return new_weights_path, predictions
    

    def run(self):
        try:
            # check if raw_dataset folder exists or not
            if not os.path.exists(self.combined_folder):
                os.makedirs(self.combined_folder+"/raw_dataset/images")
                os.makedirs(self.combined_folder+"/raw_dataset/labels")
            # check for existence of json file
            if not os.path.exists(self.json_file):
                with open(self.json_file, "w") as f:
                    json_data = {
                        "candidate_labels": []
                    }
                    json.dump(json_data, f, indent=4)
            #get camera index
            cam = AvailableCam(json_file=self.json_file)
            cam.select_camera()

            # get previous data
            if not self.new_weights:
                self.prev_data()

            # give generic name of object to detect
            object = input("What object you want to detect: \n") + "."
            # update the json file with new class
            with open(self.json_file, 'r') as file:
                data = json.load(file)
            data['candidate_labels'].append(object)
            with open(self.json_file, 'w') as file:
                json.dump(data, file, indent=4)
            
            # Create new data for object specified; and train it and get the MaP50 score
            object_specific = input("What name do you want to give to your trained object: \n")
            new_weights_path, predictions = self.new_data(object=object, object_specific=object_specific)
            return new_weights_path, predictions

        except Exception as e:
            logger.info("\n Process interrupted!")
            logger.info(e)
            if not os.listdir(f"{self.combined_folder}/raw_dataset/images"):
                shutil.rmtree(self.combined_folder)
        except KeyboardInterrupt:
            logger.info("Process interrupted in between")
            if not os.listdir(f"{self.combined_folder}/raw_dataset/images"):
                shutil.rmtree(self.combined_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Automated labelling and training')
    parser.add_argument('--new_weights', action='store_true',
                        help='Whether to create weights only for the new object')
    parser.add_argument('--prev_data_folder', type=str, required=False,
                        help='Previous dataset folder to access for training')
    parser.add_argument('--roboflow', action='store_true',
                        help='If the previous dataset is exported from Roboflow')
    parser.add_argument('--abs_yaml_file', type=str, required=False,
                        help='Absolute path to YAML file to the previous dataset (if not exported from roboflow)')
    parser.add_argument('--image_threshold', type=int, default=100, required=False,
                        help='Number of images to capture for training')
    parser.add_argument('--number_aug', type=int, default=3, required=False,
                        help='Number of augmentations to apply on the training images')
    parser.add_argument('--epochs', type=int, default=69, required=False,
                        help='Number of epochs to use for training')
    parser.add_argument('--map_threshold', type=float, default=0.5, required=False,
                        help='MAP threshold to check the output result')
    parser.add_argument('--inference', action='store_true',
                        help='To run inference on the camera added')
    parser.add_argument('--inference_threshold', type=float, default=0.3, required=False,
                        help='Confidence threshold of the inference')

    args = parser.parse_args()
    
    at = AutoTrain(new_weights=args.new_weights, prev_data_folder=args.prev_data_folder, roboflow=args.roboflow, abs_yaml_file=args.abs_yaml_file, image_threshold=args.image_threshold, number_aug=args.number_aug, epochs=args.epochs, map_threshold=args.map_threshold, inference=args.inference, inference_threshold=args.inference_threshold)
    new_weights_path, predictions = at.run()
    if predictions==None:
        logger.error("! Use more images and epochs !")

    if new_weights_path!=None:
        print("*"*20,f"\nNew weights stored in {new_weights_path}\n", "*"*20,)
