from sklearn.utils import shuffle
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,precision_recall_curve




# used for the MIL training
def epoch_train(bag_ins_list, optimizer, criterion, milnet, args):
    epoch_loss = 0
    for i, data in enumerate(bag_ins_list):
        optimizer.zero_grad()
        data_bag_list = shuffle(data[1])
        data_tensor = torch.from_numpy(np.stack(data_bag_list)).float().cuda()
        data_tensor = data_tensor[:, 0:args.num_feats]
        label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().cuda()
        classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
        max_prediction, index = torch.max(classes, 0)
        loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
        loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
        loss_total = args.loss_weight_bag*loss_bag + args.loss_weight_ins*loss_max
        loss_total = loss_total.mean()
        loss_total.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss_total.item()
    return epoch_loss / len(bag_ins_list)

def epoch_test(bag_ins_list, criterion, milnet, args):
    bag_labels = []
    bag_predictions = []
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(bag_ins_list):
            bag_labels.append(np.clip(data[0], 0, 1))
            data_tensor = torch.from_numpy(np.stack(data[1])).float().cuda()
            data_tensor = data_tensor[:, 0:args.num_feats]
            label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().cuda()
            classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
            max_prediction, index = torch.max(classes, 0)
            loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
            loss_total = 0.5*loss_bag + 0.5*loss_max
            loss_total = loss_total.mean()
            bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())
            epoch_loss = epoch_loss + loss_total.item()
    epoch_loss = epoch_loss / len(bag_ins_list)
    return epoch_loss, bag_labels, bag_predictions

def optimal_thresh(fpr, tpr, thresholds, p=0):
    """Optimal threshold for computing the possible max F1 score"""
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions):
    """
    For evaluating the model performance
    """
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore

def compute_pos_weight(bags_list):
    "Compute the positive weight"
    pos_count = 0
    for item in bags_list:
        pos_count = pos_count + np.clip(item[0], 0, 1)
    return (len(bags_list)-pos_count)/pos_count
