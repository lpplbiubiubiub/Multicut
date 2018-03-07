import numpy as np
import os
import sys
import pickle
sys.path.append("../../config")
sys.path.append("/home/xksj/workspace/lp/multi_cut/track")
from config import P
from box import Box


class MotDetParser():
    """
    parse detect txt
    """
    def __init__(self, det_file, det_th=0.):
        self._det_file = det_file
        self._det_scr_th = det_th
        assert os.path.isfile(det_file), "{} dosen't exists".format(det_file)
        self._det_dat = np.loadtxt(det_file, delimiter=",")
        conf_list = self._det_dat[:, -4]
        self._src_det_dat = self._det_dat[conf_list > self._det_scr_th]


    def get_all_det(self):
        return self._src_det_dat.copy()

    def get_by_index(self, idx):
        identical_idx = idx == self._src_det_dat[:, 0]
        return self._src_det_dat[identical_idx, :]

    def get_class(self, class_idx):
        pass

    def get_frame_idx_list(self):
        return np.sort(np.unique(self._src_det_dat[:, 0]).astype(np.int32))


class MotGtParser(object):
    """
    parse mot truth
    """
    def __init__(self, gt_file):
        assert os.path.exists(gt_file), "{} doesn't exist".format(gt_file)
        self._gt_file = gt_file
        self._gt_data = np.loadtxt(gt_file, delimiter=",", dtype=np.int32)

    def get_frame_idx(self, idx):
        identical_idx = idx == self._gt_data[:, 0]
        return self._gt_data[identical_idx]

    def get_frame_idx_cls(self, idx, cls_idx):
        identical_idx = None
        if type(cls_idx) == list or type(cls_idx) == tuple:
            for cls_item in cls_idx:
                if identical_idx is not None:
                    identical_idx += (cls_item == self._gt_data[:, -2])
                else:
                    identical_idx = (cls_item == self._gt_data[:, -2])
        else:
            assert cls_idx in P["gt_class_idx"], "Unknow class, should given (1-12)"
            identical_idx = cls_idx == self._gt_data[:, -2]
        identical_idx *= (idx == self._gt_data[:, 0])
        return self._gt_data[identical_idx]

    def get_identity_idx(self, idx):
        identical_idx = idx == self._gt_data[:, 1]
        return self._gt_data[identical_idx]

    def get_spec_class(self, cls_idx):
        identical_idx = None
        if type(cls_idx) == list or type(cls_idx) == tuple:
            for cls_item in cls_idx:
                identical_idx += (cls_item == self._gt_data[:, -2])
        else:
            assert cls_idx in P["gt_class_idx"], "Unknow class, should given (1-12)"
            identical_idx = cls_idx == self._gt_data[:, -2]
        return self._gt_data[identical_idx]


class DeepMatchingDictParser(object):
    def __init__(self, parse_dict_file):
        assert os.path.exists(parse_dict_file), "{} should exist".format(parse_dict_file)
        self._parser_dict = None
        with open(parse_dict_file, "rb") as f:
            self._parser_dict = pickle.load(f)
        self._score_thresh = 0.

    def get_res_spec_pair(self, idx_pair):
        assert type(idx_pair) == tuple, "param's type is not tuple"
        if idx_pair[1] < idx_pair[0]:
            tmp = idx_pair[0]
            idx_pair[0] = idx_pair[1]
            idx_pair[1] = tmp

        search_str = str(idx_pair[0]) + "-" + str(idx_pair[1])
        if self._parser_dict.has_key(search_str):
            return self._proprecess(self._parser_dict[search_str], self._score_thresh)

    def get_matching_iou(self, idx_pair, det1, det2):
        """
        :param idx_pair: (i1, i2)
        :param det1: x1, y1, w, h
        :param det2: x1, y1, w, h
        :return: 
        """
        match_res = self.get_res_spec_pair(idx_pair)
        return DeepMatchingDictParser.get_matching_iou_static(match_res, det1. det2)

    def _proprecess(self, match_res, score_thresh):
        """
        :param score_thresh: 
        :return: 
        """
        return match_res[match_res[:, 4] >= score_thresh]


    @staticmethod
    def get_matching_pot(mat_res, det):
        match_res = mat_res
        det1_match = (match_res[:, 0] >= det[0]) * (match_res[:, 0] <= det[0] + det[2]) * \
                    (match_res[:, 1] >= det[1]) * (match_res[:, 1] <= det[1] + det[3])
        return mat_res[det1_match]


    @staticmethod
    def get_matching_iou_static(mat_res, det1, det2, same_frm_id=False):
        match_res = mat_res
        if not same_frm_id:
            det1_match = (match_res[:, 0] >= det1[0]) * (match_res[:, 0] <= det1[0] + det1[2]) * \
                         (match_res[:, 1] >= det1[1]) * (match_res[:, 1] <= det1[1] + det1[3])
            det2_match = (match_res[:, 2] >= det2[0]) * (match_res[:, 2] <= det2[0] + det2[2]) * \
                         (match_res[:, 3] >= det2[1]) * (match_res[:, 3] <= det1[1] + det2[3])
        else:
            det1_match = (match_res[:, 0] >= det1[0]) * (match_res[:, 0] <= det1[0] + det1[2]) * \
                         (match_res[:, 1] >= det1[1]) * (match_res[:, 1] <= det1[1] + det1[3])
            det2_match = (match_res[:, 0] >= det2[0]) * (match_res[:, 0] <= det2[0] + det2[2]) * \
                         (match_res[:, 1] >= det2[1]) * (match_res[:, 1] <= det1[1] + det2[3])

        det_intersection = det1_match * det2_match
        if np.sum(det1_match) + np.sum(det2_match) - np.sum(det_intersection) == 0:
            return 0.
        matching_iou = np.sum(det_intersection) / (
            np.sum(det1_match) + np.sum(det2_match) - np.sum(det_intersection) + 0.)
        return matching_iou





if __name__ == "__main__":
    """
    mot_seq = "MOT16-02"
    det_file = os.path.join(P["mot16_data"], "train", mot_seq,
                            "det", "det.txt")
    det_dat = MotDetParser(det_file, 0.)
    # print det_dat.get_all_det()
    deep_parser = DeepMatchingDictParser(os.path.join(P['DeepMatchingResDir'], "MOT16-02.pth"))
    res = deep_parser.get_res_spec_pair((100, 104))
    # print(res.shape)
    res = np.sort(res, axis=0)
    for i in range(res.shape[0]):
        print(int(res[i][0]))
    iou = deep_parser.get_matching_iou((413, 421), (912,484,97,109), (912,484,97,109))
    print(iou)
    """
    """
    mot_seq = "MOT16-02"
    det_file = os.path.join(P["mot16_data"], "train", mot_seq,
                            "gt", "gt.txt")
    gt_parser = MotGtParser(det_file)
    print gt_parser.get_frame_idx(421)
    """
    # visual deepmatching
    import cv2
    mot_seq = "MOT16-13"
    det_file = os.path.join(P["mot16_data"], "train", mot_seq,
                            "det", "det.txt")
    det_parser = MotDetParser(det_file, 0.)
    deep_parser = DeepMatchingDictParser(os.path.join(P['DeepMatchingResDir'], "train", mot_seq +  ".pth"))
    gt_file = os.path.join(P["mot16_data"], "train", mot_seq,
                          "gt", "gt.txt")
    gt_parser = MotGtParser(gt_file)

    frm_id1, frm_id2 = 420, 430
    det1_arr = det_parser.get_by_index(frm_id1)
    det2_arr = det_parser.get_by_index(frm_id2)
    print("shape is ---> ", det1_arr.shape, det2_arr.shape)
    gt1_arr = gt_parser.get_frame_idx_cls(frm_id1, P['gt_ped_class_idx'])
    gt2_arr = gt_parser.get_frame_idx_cls(frm_id2, P['gt_ped_class_idx'])

    res = deep_parser.get_res_spec_pair((frm_id1, frm_id2))
    img_dir = os.path.join(P["mot16_data"], "train", mot_seq,
                            "img1")
    img1_file = os.path.join(img_dir, str(frm_id1).zfill(6) + ".jpg")
    img2_file = os.path.join(img_dir, str(frm_id2).zfill(6) + ".jpg")
    assert os.path.exists(img1_file), "img1_file {}".format(img1_file)
    assert os.path.exists(img2_file), "img2_file {}".format(img2_file)
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    # draw circles
    random_color_arr = np.random.randint(0, 255, size=(res.shape[0], 3))
    rec_random_color_arr = np.random.randint(0, 255, size=(res.shape[0], 3))

    [cv2.circle(img1, (res[i][0], res[i][1]), color=random_color_arr[i], radius=2, thickness=2) for i in range(res.shape[0])]
    [cv2.circle(img2, (res[i][2], res[i][3]), color=random_color_arr[i], radius=2, thickness=2) for i in range(res.shape[0])]

    for i in range(det1_arr.shape[0]):
        cv2.rectangle(img1, (int(det1_arr[i][2]), int(det1_arr[i][3])), (int(det1_arr[i][4]) + int(det1_arr[i][2]),
                                                                         int(det1_arr[i][5]) + int(det1_arr[i][3])), color=(255, 0, 0), thickness=3)
        box = Box(pos=det1_arr[i][2:6])
        box.get_gt_id(gt1_arr)
        cv2.putText(img1, str(box.cls), color=(0, 0, 255), org=(int(det1_arr[i][2]), int(det1_arr[i][3])), fontScale=2, fontFace=2, thickness=3)
        cv2.putText(img1, str(int(box.pos[0])), color=(0, 0, 255),
                    org=(int(det1_arr[i][2]),
                         int(det1_arr[i][3]) + 40), fontScale=1, fontFace=2, thickness=2)

    for i in range(det2_arr.shape[0]):
        cv2.rectangle(img2, (int(det2_arr[i][2]), int(det2_arr[i][3])), (int(det2_arr[i][4]) + int(det2_arr[i][2]),
                                                                         int(det2_arr[i][5]) + int(det2_arr[i][3])), color=(255, 0, 0), thickness=3)
        box = Box(pos=det2_arr[i][2:6])
        box.get_gt_id(gt2_arr)
        cv2.putText(img2, str(box.cls), color=(0, 0, 255), org=(int(det2_arr[i][2]), int(det2_arr[i][3])), fontScale=2, fontFace=2, thickness=3)
        cv2.putText(img2, str(int(box.pos[0])), color=(0, 0, 255), org=(int(det2_arr[i][2]),
                                                                        int(det2_arr[i][3]) + 40), fontScale=1, fontFace=2, thickness=2)

    for i in range(det1_arr.shape[0]):
        for ii in range(det2_arr.shape[0]):
            box1 = Box(pos=det1_arr[i][2:6])
            box1.get_gt_id(gt1_arr)

            box2 = Box(pos=det2_arr[ii][2:6])
            box2.get_gt_id(gt2_arr)
            print(box1.pos, box2.pos)
            print(box1.cls, box2.cls)
            print("iou", deep_parser.get_matching_iou_static(res, box1.pos, box2.pos))

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)


