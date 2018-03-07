"""
About cls
-2 means undefined, -1 means background
"""
import sys
import numpy as np
import sys
sys.path.append("../config")
from config import P
BACKGROUND_CLS_ID = -1


class Box(object):
    def __init__(self, pos, frm_id=0, conf=0., cls=-2, uid=-1):
        """
        :param pos: x, y, w, h
        :param conf: 
        :param cls: 
        """

        self._pos = Box.init_pos(pos)
        self._conf = conf
        self._cls = cls
        self._frm_id = frm_id
        self._uid = uid

    def get_gt_id(self, gt_arr):
        if gt_arr is None:
            return
        assert type(gt_arr) is np.ndarray, "type of gt_arr should be np arr"
        iou_res_list = np.array([self.box_iou(gt_arr[i][2:6]) for i in range(gt_arr.shape[0])])
        if len(iou_res_list) == 0:
            return
        max_idx = np.argmax(iou_res_list)
        if iou_res_list[max_idx] > P["det_match_thresh"]:
            self._cls = int(gt_arr[max_idx][1])
        else:
            self._cls = BACKGROUND_CLS_ID

    def box_iou(self, det):
        """
        :param det: x, y, w, h
        :return: 
        """
        box2 = Box(pos=(det[0], det[1], det[2], det[3]))
        return Box.iou(self, box2)

    @staticmethod
    def iou(box1, box2):
        assert type(box1) is Box, "box1 is not Box"
        assert type(box2) is Box, "box2 is not Box"
        pos1 = (box1.pos[0], box1.pos[1], box1.pos[0] + box1.pos[2], box1.pos[1] + box1.pos[3])
        pos2 = (box2.pos[0], box2.pos[1], box2.pos[0] + box2.pos[2], box2.pos[1] + box2.pos[3])

        iw = max(0, (min(pos1[2], pos2[2]) - max(pos1[0], pos2[0])))
        ih = max(0, (min(pos1[3], pos2[3]) - max(pos1[1], pos2[1])))
        intersect = iw * ih
        union = box1.area + box2.area - intersect
        if union < 1e-6:
            return 0.
        else:
            return float(intersect) / union

    @staticmethod
    def init_pos(pos):
        x, y, w, h = pos[0], pos[1], pos[2], pos[3]
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        return x, y, w, h

    @property
    def pos(self):
        return self._pos

    @property
    def conf(self):
        return self._conf

    @property
    def cls(self):
        return self._cls

    @property
    def area(self):
        return self.width * self.height

    @property
    def x(self):
        return self._pos[0]

    @property
    def y(self):
        return self._pos[1]
    @property
    def width(self):
        return self._pos[2]

    @property
    def height(self):
        return self._pos[3]

    @property
    def frm_id(self):
        return self._frm_id

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, uid_num):
        self._uid = uid_num