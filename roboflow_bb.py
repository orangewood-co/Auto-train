import os
import shutil
import cv2
import yaml
import json

'''
DRAW BB ON IMAGES GOT FROM ROBOFLOW; COMBINE THE THREE SUBFOLDERS INTO ONE DATASET AND USE THAT FOR SPLITTING AHEAD
[THIS IS JUST FOR FIRST WEIGHTS, SO THAT COULD BE COMBINED WITH THE NEW ONES]
INPUT: ROBOFLOW TXT AND IMAGES
OUTPUT: ANNOTATED IMAGES ALONG WITH TXT(SAME AS INPUT) ALL IN A SINGLE DATASET
'''

class RoboflowBB:
    def __init__(self, prev_folder, combined_folder, json_file, abs_yaml_file):
        self.prev_folder = prev_folder
        self.combined_folder = combined_folder+"/raw_dataset"
        self.json_file = json_file
        self.abs_yaml_file = abs_yaml_file

    def make_copy_folder(self):
        '''
        folder = path to upper folder containing splitted subfolders
        '''
        for subfolder in os.listdir(self.prev_folder):
            if os.path.isdir(f"{self.prev_folder}/{subfolder}/"):
                for item in os.listdir(f"{self.prev_folder}/{subfolder}/images"):
                    source_path = os.path.join(f"{self.prev_folder}/{subfolder}/images", item)
                    destination_path = os.path.join(f"{self.combined_folder}/images", item)
                    shutil.copy(source_path,destination_path)
                    # print(f"Done for images in {subfolder}")
                for item in os.listdir(f"{self.prev_folder}/{subfolder}/labels"):
                    source_path = os.path.join(f"{self.prev_folder}/{subfolder}/labels", item)
                    destination_path = os.path.join(f"{self.combined_folder}/labels", item)
                    shutil.copy(source_path,destination_path)
                    # print(f"Done for labels in {subfolder}")


    def draw_bb(self):
        '''
        folder = path to combined folder containing images, labels
        '''
        for image in os.listdir(f"{self.combined_folder}/images"):
            image_path = f"{self.combined_folder}/images/{image}"
            image_cv = cv2.imread(image_path)
            ih,iw = image_cv.shape[:2]
            initials = os.path.splitext(image)[0]
            label = f"{initials}.txt"
            if label in os.listdir(f"{self.combined_folder}/labels"):
                with open(f"{self.combined_folder}/labels/{label}", "r") as fl:
                    bb_list = []
                    label_content = fl.read()
                    label_content = label_content.split("\n")
                    for labels in label_content:
                        if len(labels)!=0:
                            xcn, ycn, wn, hn = [float(i) for i in labels.split(" ")[1:]]
                            xc,yc,w,h = xcn*iw, ycn*ih, wn*iw, hn*ih
                            xmax, xmin, ymax, ymin = int((2*xc+w)/2), int((2*xc-w)/2), int((2*yc+h)/2), int((2*yc-h)/2)
                            bb_ind_list = [xmin, ymin, xmax, ymax]
                        bb_list.append(bb_ind_list)
                    for bb in bb_list:
                        image_sh = cv2.rectangle(image_cv, (bb[0],bb[1]), (bb[2],bb[3]), (0,255,0), 2)
                    
                    cv2.imwrite(image_path,image_sh)
    
    def update_json_from_yaml(self):
        '''
        If the subfolder contains the yaml file, then use its classes to update the json
        '''
        if self.abs_yaml_file==None:
            yaml_files = [file for file in os.listdir(self.prev_folder) if file.endswith('.yaml') or file.endswith('.yml')]
            yaml_file = yaml_files[0]
            if not yaml_files:
                raise ValueError("yaml file must be in same directory as previous data for roboflow=True")
            else:
                with open(f"{self.prev_folder}/{yaml_file}", 'r') as file:
                    yaml_data = yaml.safe_load(file)
                names_list = yaml_data.get('names', [])
                with open(self.json_file, 'r') as file:
                    data = json.load(file)
                data['candidate_labels']=names_list
                with open(self.json_file, 'w') as file:
                    json.dump(data, file, indent=4)
        else:
            with open(self.abs_yaml_file, 'r') as file:
                yaml_data = yaml.safe_load(file)
            names_list = yaml_data.get('names', [])
            if type(names_list) is dict:
                with open(self.json_file, 'r') as file:
                    data = json.load(file)
                data['candidate_labels']=list(names_list.values())
                with open(self.json_file, 'w') as file:
                    json.dump(data, file, indent=4)
            else:
                with open(self.json_file, 'r') as file:
                    data = json.load(file)
                data['candidate_labels']=names_list
                with open(self.json_file, 'w') as file:
                    json.dump(data, file, indent=4)

    
    def run(self):
        self.make_copy_folder()
        self.draw_bb()
        print("Stored combined data \n")
        self.update_json_from_yaml()
        print("We have updated json file now \n")
