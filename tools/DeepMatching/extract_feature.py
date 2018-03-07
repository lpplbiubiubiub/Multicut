import sys
sys.path.append("/home/xksj/workspace/lp/DeepMatchingGPU")
sys.path.append("../../config")
from config import P
import os
import cv2

# import deep_matching_gpu.match_images_wrapper_v2 as match_images_warpper
from deep_matching_gpu import match_images_wrapper_v2
from deep_matching_gpu import match_images_warpper
import pickle
match_images_warpper = match_images_wrapper_v2

def extract_match_info(img_dir="", match_gap=10, save_feature_file=""):
    """
    :param img_dir:
     the image dir you want to match images
    :param match_gap: 
     the max gap between image timestamp
    :param save_feature_dir: 
     the feature save dir
    :return: 
    """
    match_dict = {}
    assert os.path.exists(img_dir), "{} doesn't exist".format(img_dir)
    img_list = os.listdir(img_dir)
    list.sort(img_list, key=lambda x: int(x.split(".")[0]))
    img_timestamp_list = [int(name.split(".")[0]) for name in img_list]
    img_full_path_list = [os.path.join(img_dir, img_file) for img_file in img_list]
    assert all(os.path.isfile(img_file) for img_file in img_full_path_list), "img should exist"
    # we just need forward match, backward match si unnecessary
    max_timestamp = max(img_timestamp_list)
    max_idx = len(img_list) - 1
    for i, img in enumerate(img_list):
        iter_end = i + 10 if (i + 10) <= max_idx else max_idx
        if i == iter_end + 1:
            continue
        for idx in range(i + 1, iter_end + 1):
            img1 = cv2.imread(img_full_path_list[i])
            img2 = cv2.imread(img_full_path_list[idx])

            match_res = match_images_warpper((img1, img2), 3, 128)[0]
            match_dict[str(img_timestamp_list[i]) + "-" + str(img_timestamp_list[idx])] = match_res

            print(match_res.shape, i, idx)

    with open(save_feature_file, "wb") as f:
        pickle.dump(match_dict, f)

if __name__ == "__main__":
    seq_list = P["train_mot_seq"]
    for seq in ("MOT16-13", ):
        extract_match_info(os.path.join(P["mot16_data"], "train/{}/img1".format(seq)), 10,
                        os.path.join(P['DeepMatchingResDir'], "train", "{}.pth".format(seq)))
        print("{} has been save".format(os.path.join(P['DeepMatchingResDir'], "{}.pth".format(seq))))



