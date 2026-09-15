import sys
sys.path.insert(0, './utils/')
from help_functions import load_hdf5
from extract_patches import pred_only_FOV
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import configparser
import h5py
import numpy as np

config = configparser.ConfigParser()
config.read('configuration.txt')

path_data = config.get('data paths', 'path_local')
dataset = config.get('data attributes', 'dataset')
name = config.get('experiment name', 'name')

path_experiment = './log/experiments/' + name + '/' + dataset + '_cross_DRIVE/'

# 读取跨域预测结果
file = h5py.File(path_experiment + dataset + '_predict_results.h5', 'r')
gtruth_masks = file['y_gt'][:]
pred_prob = file['y_prob'][:]
file.close()

# 读取 border masks
test_border_masks = load_hdf5(path_data + config.get('data paths', 'test_border_masks'))

# 只在 FOV 内评估
y_scores, y_true = pred_only_FOV(pred_prob, gtruth_masks, test_border_masks, insideFOV=True)
y_true = np.asarray(y_true)
if np.max(y_true) > 1:
    y_true = y_true // np.max(y_true)
y_true = y_true.astype(np.int64)

# AUC（用概率分数，不受阈值影响）
AUC_ROC = roc_auc_score(y_true, y_scores)

print("=" * 60)
print("Cross-dataset result (DRIVE weight -> %s)" % dataset)
print("AUC (ROC): %.4f" % AUC_ROC)
print("=" * 60)
print("%-10s %-8s %-8s %-8s %-8s" % ("threshold", "SE", "SP", "ACC", "F1"))
for th in [0.3, 0.4, 0.5, 0.6]:
    y_pred = (y_scores >= th).astype(np.int64)
    confusion = confusion_matrix(y_true, y_pred)
    tn = confusion[0, 0]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    tp = confusion[1, 1]
    accuracy = float(tp + tn) / float(tp + tn + fp + fn)
    specificity = float(tn) / float(tn + fp)
    sensitivity = float(tp) / float(tp + fn)
    F1 = f1_score(y_true, y_pred)
    print("%-10.1f %-8.4f %-8.4f %-8.4f %-8.4f" % (th, sensitivity, specificity, accuracy, F1))
print("=" * 60)
