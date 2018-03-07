"""
This file should collect some feature from detection file and gt file, then out put feature file for train;
first, it collect det info with gt id,
then, we collect DeepMatching info from pair det.
finally, we wrap feature to a file
"""
import numpy as np
import sys
sys.path.append("../../track")
sys.path.append("../../tools/Parser")
sys.path.append("../../config")
from tools import MotDetParser
from tools import MotGtParser
from tools import DeepMatchingDictParser
from config import P
from track import Box
import os
import struct



def pair_feature(box1, box2, **kwargs):
    """
    Feature for mat is (deepmatching iou(f1), min_conf(f2), f1*f2, f1**2, f2**2)
    :param box1: 
    :param box2: 
    :param kwargs: pair feature calculated by outside 
    :return: 
    """
    assert type(box1) is Box and type(box2) is Box, "box1 and box2 should have the Box type"
    param_dict = kwargs
    f1 = param_dict["deep_matching"]
    f2 = min(box1.conf, box2.conf)
    f3 = f1 * f2
    f4 = f1 ** 2
    f5 = f2 ** 2
    same_flag = 1 if box1.cls == box2.cls and box1.cls > 0 else 0
    return np.array((f1, f2, f3, f4, f5)), same_flag


def st_pair_feature(box1, box2, **kwargs):
    assert type(box1) is Box and type(box2) is Box, "box1 and box2 should have the Box type"
    h_ = abs(box1.height + box2.height) / 2.
    f1 = abs(box1.frm_id - box2.frm_id)
    f2 = abs(box1.x - box2.x) / h_
    f3 = abs(box1.y - box2.y) / h_
    f4 = Box.iou(box1, box2)
    f5 = min(box1.conf, box2.conf)
    f6 = abs(box1.height - box2.height) / h_
    same_flag = 1 if box1.cls == box2.cls and box1.cls > 0 else 0
    return np.array((f1, f2, f3, f4, f5, f6)), same_flag

class TrainDataProcess(object):
    """
    process info to train data    
    """
    def __init__(self, mot_seq, use_st=False, is_test=False, **kwargs):
        """
        :param mot_seq: like "MOT16-02"
        :param use_st: 
        :param is_test: if for test seq, then there is no gt file
        :param coef: logistic coef
        """
        self._mot_seq = mot_seq
        flag = "train"
        self._test_flag = is_test

        if is_test:
            flag = "test"
            self._coef = kwargs['coef']
            self._mean = kwargs['mean']
            self._std = kwargs['std']
            if use_st:
                self._multicut_ver_file = os.path.join(P['MulticutFormatDir'], mot_seq + "_st_vertex.dat")
                self._multicut_weight_edge_file = os.path.join(P['MulticutFormatDir'], mot_seq + "_st_edge.dat")
            else:
                self._multicut_ver_file = os.path.join(P['MulticutFormatDir'], mot_seq + "_vertex.dat")
                self._multicut_weight_edge_file = os.path.join(P['MulticutFormatDir'], mot_seq + "_edge.dat")


        gt_file = os.path.join(P["mot16_data"], flag, mot_seq,
                               "gt", "gt.txt")
        det_file = os.path.join(P["mot16_data"], flag, mot_seq,
                                "det", "det.txt")
        self.det_parser = MotDetParser(det_file, 0.)
        if not is_test:
            self.gt_parser = MotGtParser(gt_file)
        else:
            self.gt_parser = None
        use_deep_matching = not use_st
        if use_st:
            self.feature_file = os.path.join(P["FeatureDir"], "ST", mot_seq + "_feature.pth")
            self.target_file = os.path.join(P["FeatureDir"], "ST", mot_seq + "_target.pth")
            self.gap_file = os.path.join(P["FeatureDir"], "ST", mot_seq + "_gap.pth")
            self._pair_feature = st_pair_feature
        else:
            self.feature_file = os.path.join(P["FeatureDir"], mot_seq + "_feature.pth")
            self.target_file = os.path.join(P["FeatureDir"], mot_seq + "_target.pth")
            self.gap_file = os.path.join(P["FeatureDir"], mot_seq + "_gap.pth")
            self._pair_feature = pair_feature
        self._use_deep_matching = use_deep_matching
        self.deep_parser = None
        if use_deep_matching:
            if not is_test:
                self.deep_parser = DeepMatchingDictParser(os.path.join(P['DeepMatchingResDir'], mot_seq + ".pth"))
            else:
                self.deep_parser = DeepMatchingDictParser(os.path.join(P['DeepMatchingResDir'], "test",  mot_seq + ".pth"))
        self.frm_idx_list = self.det_parser.get_frame_idx_list()
        self._box_dict = {}  # {key: val = frame_idx: box_list}

    def process_box(self):
        """
        :return: 
        """
        det_dat = self.det_parser.get_all_det()
        # box --> get gt it --> gt deep matching --> get feature
        temp_box_pool = []
        temp_idx_pool = []
        X_list = []
        y_list = []
        gap_list = []
        vertice_info_list = []
        edge_info_list = []
        box_uid = 0  # assign box a unique id
        for frm_id_idx in range(self.frm_idx_list.shape[0]):
            box_list = []
            frm_id = self.frm_idx_list[frm_id_idx]
            gt_arr = None

            if self.gt_parser:
                gt_arr = self.gt_parser.get_frame_idx_cls(frm_id, P['gt_ped_class_idx'])
            print(frm_id)
            if len(temp_idx_pool) >= P["deep_matching_size"]:
                del_box_id = temp_idx_pool[0] # pop idx
                del temp_idx_pool[0]
                del_idx_list = [] # pop box
                for idx, box in enumerate(temp_box_pool):
                    if box.frm_id == del_box_id:
                        del_idx_list.append(idx)
                for idx, del_idx in enumerate(del_idx_list):
                    del temp_box_pool[del_idx - idx]

            det_arr = self.det_parser.get_by_index(frm_id)
            same_deep_matching_res = None
            if self.deep_parser:
                same_deep_matching_res = self.deep_parser.get_res_spec_pair((frm_id, frm_id + 1))

            if same_deep_matching_res is not None or self.deep_parser is None:
                for idx in range(det_arr.shape[0]):
                    det = det_arr[idx]
                    tmp_box = Box(pos=det[2:6], frm_id=det[0], conf=det[-4], uid=box_uid)
                    tmp_box.get_gt_id(gt_arr)
                    # assigned id
                    box_list.append(tmp_box)
                    if self._test_flag:
                        vertice_info_list.append((frm_id, box_uid, int(det[2]), int(det[3]), int(det[4]), int(det[5])))
                    box_uid += 1
                
            ###########################################
            # cal same frm feature

            if same_deep_matching_res is not None or self.deep_parser is None:
                for i in range(len(box_list) - 1):
                    for ii in range(i, len(box_list)):
                        box1, box2 = box_list[i], box_list[ii]
                        if self.deep_parser:
                            deep_matching_fea = DeepMatchingDictParser.get_matching_iou_static(same_deep_matching_res, box1.pos, box2.pos)
                            x, y = self._pair_feature(box1, box2, deep_matching=deep_matching_fea)
                        else:
                            x, y = self._pair_feature(box1, box2)
                        gap = abs(box1.frm_id - box2.frm_id)
                        X_list.append(x)
                        y_list.append(y)
                        gap_list.append(gap)
                        if self._test_flag:
                            # print("pair is {}".format(np.matmul(self._coef, x)))
                            # edge_info_list.append((box1.uid, box2.uid, np.matmul(self._coef, (x - self._mean) / self._std)))
                            edge_info_list.append(
                                 (box1.uid, box2.uid, np.matmul(self._coef, x)))
            # cal diff frm feature

            ###########################################
            for tmp_frm_id in temp_idx_pool:
                deep_matching_res = None
                if self.deep_parser:
                    deep_matching_res = self.deep_parser.get_res_spec_pair((frm_id, frm_id + 1))
                if deep_matching_res is not None or self.deep_parser is None:
                    for pre_box in temp_box_pool:
                        if pre_box.frm_id == tmp_frm_id:
                            for tmp_box in box_list:
                                box1, box2 = pre_box, tmp_box
                                if self.deep_parser:
                                    deep_matching_fea = DeepMatchingDictParser.get_matching_iou_static(
                                        deep_matching_res, box1.pos, box2.pos)
                                    x, y = self._pair_feature(box1, box2, deep_matching=deep_matching_fea)
                                else:
                                    x, y = self._pair_feature(box1, box2)
                                X_list.append(x)
                                y_list.append(y)
                                gap = abs(box1.frm_id - box2.frm_id)
                                gap_list.append(gap)
                                if self._test_flag:
                                    # print("pair is {}".format(np.matmul(self._coef, x)))
                                    # edge_info_list.append(
                                    #     (box1.uid, box2.uid, np.matmul(self._coef, (x - self._mean) / self._std)))
                                    edge_info_list.append(
                                             (box1.uid, box2.uid, np.matmul(self._coef, x)))

            # aft cal pair feature
            temp_idx_pool.append(frm_id)
            temp_box_pool.extend(box_list)
        X_arr = np.array(X_list)
        y_arr = np.array(y_list)
        gap_arr = np.array(gap_list)
        # save file
        np.save(open(self.feature_file, "wb"), X_arr)
        np.save(open(self.target_file, "wb"), y_arr)
        np.save(open(self.gap_file, "wb"), gap_arr)
        if self._test_flag:
            with open(self._multicut_ver_file, "wb") as f:
                data = struct.pack("i", len(vertice_info_list))
                f.write(data)
                [f.write(struct.pack("6i", *vertex)) for vertex in vertice_info_list]
                for vertex in vertice_info_list:
                    print(vertex)
            with open(self._multicut_weight_edge_file, "wb") as f:
                data = struct.pack("i", len(edge_info_list))
                f.write(data)
                [f.write(struct.pack("2if", int(weight_edge[0]), int(weight_edge[1]),
                                     float(weight_edge[2]))) for weight_edge in edge_info_list]


if __name__ == "__main__":
    seq_list = P["train_mot_seq"]
    for i, mot_seq in enumerate(("MOT16-13", )):
        train_seq = P['train_mot_seq'][i]
        print(mot_seq, train_seq)
        weight_param = np.loadtxt(os.path.join(P["TrainedModelDir"], train_seq + "_logistic.model"))
        coef = weight_param[0]
        mean = weight_param[1]
        std = weight_param[2]
        t = TrainDataProcess(mot_seq, use_st=False, is_test=False, coef=coef, mean=mean, std=std)
        t.process_box()

