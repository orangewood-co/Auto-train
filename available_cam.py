import cv2
import json

class AvailableCam():
    def __init__(self, json_file, range=5):
        self.range = range
        self.json_file = json_file

    def get_available_cameras(self):
        available_cameras = []
        # Check for cameras 
        for i in range(self.range):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def select_camera(self):
        cameras = self.get_available_cameras()
        #check for present cameras
        if cameras:
            #for multiple cameras
            if len(cameras)>1:
                print("Available Cameras:", cameras)
                while True:
                    cam_index = input('Enter camera index to use: ')
                    if int(cam_index) in cameras:
                        #store the cam index as dict in input.json
                        with open(self.json_file, 'r') as file:
                            data = json.load(file)
                        data['camera_index'] = cam_index
                        with open(self.json_file, 'w') as file:
                            json.dump(data, file, indent=4)
                        print(f'Camera accessed: {cam_index}')
                        break
                    else:
                        print('Choose the camera from the indexes given above')
            else:
                #store the cam index as dict in input.json
                with open(self.json_file, 'r') as file:
                    data = json.load(file)
                data['camera_index'] = cameras[0]
                print(f'Camera accessed: {cameras[0]}')
                with open(self.json_file, 'w') as file:
                    json.dump(data, file, indent=4)
        else:
            print("No cameras found.")