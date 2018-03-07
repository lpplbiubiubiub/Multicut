import numpy as np
from tools import LogisticTrainer
from config import P
import os

if __name__ == "__main__":
    # load data
    use_st = False
    mot_list = P["train_mot_seq"]
    for i, mot_seq in enumerate(mot_list):
        print(mot_seq)
        if use_st:
            feature_file = os.path.join(P["FeatureDir"], "ST",  mot_seq + "_feature.pth")
            target_file = os.path.join(P["FeatureDir"], "ST", mot_seq + "_target.pth")
            gap_file = os.path.join(P["FeatureDir"], "ST", mot_seq + "_gap.pth")
        else:
            feature_file = os.path.join(P["FeatureDir"], mot_seq + "_feature.pth")
            target_file = os.path.join(P["FeatureDir"], mot_seq + "_target.pth")
            gap_file = os.path.join(P["FeatureDir"], mot_seq + "_gap.pth")
        X_arr = np.load(feature_file)
        y_arr = np.load(target_file)
        gap_arr = np.load(gap_file)
        log_train = LogisticTrainer(X=X_arr, Y=y_arr, iter=1000, test_size=0.2)
        log_train.fit(balance_sample=True)
        log_train.val()
        # coef, mean, std = np.loadtxt(os.path.join(P['TrainedModelDir'], "MOT16-02" + "_logistic.model"))
        # log_train.guest_param_val(coef, mean, std)
        if use_st:
            log_train.save_model(os.path.join(P['TrainedModelDir'], mot_seq + "_st_logistic.model"))
        else:
            log_train.save_model(os.path.join(P['TrainedModelDir'], mot_seq + "_logistic.model"))



