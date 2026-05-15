
import time
import yaml
import numpy as np

from teleop.TeleVision import OpenTeleVision
from teleop.Preprocessor import VuerPreprocessor
from multiprocessing import shared_memory, Queue,Event


class OpenTeleVisionOps():
    def __init__(self):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

    def get_data_from_open_tele_vision(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand, right_hand = self.processor.process(self.tv)

        left_hand_mat = np.tile(np.eye(4), (25, 1, 1))
        right_hand_mat = np.tile(np.eye(4), (25, 1, 1))
        left_hand_mat[:, :3, 3] = left_hand
        right_hand_mat[:, :3, 3] = right_hand

        left_hand_mat = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,-1,0],[0,0,0,1]])@left_hand_mat
        right_hand_mat = np.array([[0,1, 0,0],[1, 0,0,0],[0,0,-1,0],[0,0,0,1]])@right_hand_mat
        # print(right_hand_mat[19,:,:])
        # print("left_hand",left_hand)
        # print(right_hand_mat)

        data = {
            'head': head_mat,
            'left_wrist': left_wrist_mat,
            'right_wrist': right_wrist_mat,
            'left_fingers': left_hand_mat,
            'right_fingers': right_hand_mat,
        }
        return data

