import cv2
import numpy as np
import time
import os
from visdom import Visdom

lk_params = dict( winSize=(15,15), # win size is the search size
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class MatchObject(object):
    """
    
    """
    def __init__(self, img_pair, detection_pair, sample_per_det=50):
        """
        img_pair:(tuple)
            contain image pair
        detection_pair:(tuple)
            contain pair of detetcion list
        """
        assert type(img_pair) == tuple and len(img_pair) == 2, "type of img pair must be tuple"
        self._img_pair = img_pair
        assert type(detection_pair) == tuple and len(detection_pair) == 2, "type of detection_pair must be pair"
        self._det_pair = detection_pair
        self._sample_per_det = sample_per_det
        self._width = img_pair[0].shape[1]
        self._height = img_pair[1].shape[0]
        np.random.seed(1)
        self._viz = Visdom()

    def match(self):
        """
        process match
        :return: 
        """
        first_dets = self._det_pair[0]
        second_dets = self._det_pair[1]
        first_match_points_list = []
        second_match_points_list = []
        first_gray_img = cv2.cvtColor(self._img_pair[0], cv2.COLOR_BGR2GRAY)
        second_gray_img = cv2.cvtColor(self._img_pair[1], cv2.COLOR_BGR2GRAY)
        first_belong_box = []
        second_belong_box = []
        """
        for i, det in enumerate(first_dets):
            x1, y1, w, h = det
            x2, y2 = x1 + w, y1 + h
            if x1 < 0:
                x1 = 0
            if x1 >= self._width:
                x1 = self._width - 1
            if x2 < 0:
                x2 = 0
            if x2 >= self._width:
                x2 = self._width - 1
            rand_x = np.random.randint(x1, x2, (self._sample_per_det, 1))
            rand_y = np.random.randint(y1, y2, (self._sample_per_det, 1))
            rand_points = np.column_stack((rand_x, rand_y))
            first_match_points_list.extend(rand_points)
            [first_belong_box.append(i) for c in range(self._sample_per_det)]
        """

        for i, det in enumerate(first_dets):
            x1, y1, w, h = det
            x2, y2 = x1 + w, y1 + h
            if x1 < 0:
                x1 = 0
            if x1 >= self._width:
                x1 = self._width - 1
            if x2 < 0:
                x2 = 0
            if x2 >= self._width:
                x2 = self._width - 1
            mask = np.zeros(shape=(self._height, self._width, 1), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            p0 = cv2.goodFeaturesToTrack(first_gray_img, mask=mask, **feature_params)
            print(p0)
            first_match_points_list.extend(p0.reshape(-1, 2).tolist())
            [first_belong_box.append(i) for c in range(p0.shape[0])]
        """
        for i, det in enumerate(second_dets):
            x1, y1, w, h = det
            x2, y2 = x1 + w, y1 + h
            if x1 < 0:
                x1 = 0
            if x1 >= self._width:
                x1 = self._width - 1
            if x2 < 0:
                x2 = 0
            if x2 >= self._width:
                x2 = self._width - 1

            rand_x = np.random.randint(x1, x2, (self._sample_per_det, 1))
            rand_y = np.random.randint(y1, y2, (self._sample_per_det, 1))
            rand_points = np.column_stack((rand_x, rand_y))
            second_match_points_list.extend(rand_points)
            [second_belong_box.append(i) for c in range(self._sample_per_det)]
        """
        # find good feature

        first_points_arr = np.array(first_match_points_list)
        # second_points_arr = np.array(second_match_points_list)
        first_belong_box = np.array(first_belong_box).reshape(-1, 1)
        # second_belong_box = np.array(second_belong_box).reshape(-1, 1)

        first_points_arr = first_points_arr.reshape(-1, 1, 2).astype(np.float32)

        p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray_img, second_gray_img, first_points_arr,
                                 None, **lk_params)
        dst_img = np.zeros(shape=(self._height, self._width * 2, 3), dtype=np.uint8)
        dst_img[:, 0:self._width, :] = self._img_pair[0][:, :, :]
        dst_img[:, self._width:, :] = self._img_pair[1][:, :, :]

        first_points_arr = first_points_arr[st==1]
        p1 = p1[st==1]
        first_belong_box = first_belong_box[st==1]
        # second_belong_box = second_belong_box[st==1]
        # draw rect
        [cv2.rectangle(dst_img, (item[0], item[1]),
                       (item[0] + item[2], item[1] + item[3]), (0,255,0),3) for item in first_dets]
        [cv2.rectangle(dst_img, (item[0] + self._width, item[1]),
                       (item[0] + item[2] + self._width, item[1] + item[3]), (0,255,0),3)
                                                for item in second_dets]


        box_unique, unique_idx = np.unique(first_belong_box, return_index=True)
        rand_color = np.random.randint(0, 255, size=(box_unique.shape[0], 3))
        for point1, point2, ids in zip(first_points_arr.reshape(-1, 2), p1.reshape(-1, 2), first_belong_box):

            cv2.circle(dst_img, (int(point1[0]), int(point1[1])), 3, (255, 255, 0), 3)
            cv2.circle(dst_img, (int(point2[0] + self._width), int(point2[1])), 3, (255, 255, 0), 3)
            cv2.line(dst_img, (int(point1[0]), int(point1[1])), (int(point2[0] + self._width), int(point2[1])), rand_color[ids], 2)

        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        dst_img = dst_img.transpose(2, 0, 1).astype(np.uint8)
        self._viz.image(dst_img, win="img")


if __name__ == "__main__":
    detection_file = "/home/xksj/Data/lp/MOT16/train/MOT16-02/det/det.txt"
    img_dir = "/home/xksj/Data/lp/MOT16/train/MOT16-02/img1"
    detection_arr = np.loadtxt(detection_file, delimiter=",")
    detection_arr = detection_arr[detection_arr[:, -4] > 0.]
    detection_idx = detection_arr[:, 0].astype(np.int32)
    detection_pos = detection_arr[:, 2:6].astype(np.int32)
    img_list = os.listdir(img_dir)
    idx_pair = (10, 20)
    img_file_list = [os.path.join(img_dir, str(i).zfill(6) + ".jpg") for i in idx_pair]
    first_det_list = detection_pos[idx_pair[0] == detection_idx]

    second_det_list = detection_pos[idx_pair[1] == detection_idx]
    first_img = cv2.imread(img_file_list[0])
    second_img = cv2.imread(img_file_list[1])
    img_pair = (first_img, second_img)
    demo = MatchObject((first_img, second_img), (first_det_list, second_det_list), 100)
    demo.match()




