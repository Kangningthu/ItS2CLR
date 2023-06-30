import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.losses import SupConLoss
import shutil
import sys
import comet_ml
from comet_ml import Experiment
from dataset.dataloader_simCLR import SSLDataset

import torch
from torchvision import models
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import sys, argparse, os, copy, itertools
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

import models.dsmil as mil

from dataset.dataloader_simCLR import BagDataset_ins as BagDataset
import pickle
from utils import epoch_train, epoch_test, optimal_thresh, five_scores, compute_pos_weight

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)

apex_support = False


def _save_config_file(model_checkpoints_folder):
    """Save the config file to the model checkpoints folder"""
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


def get_features(net, dataset):
    """
    get the extracted feature for MIL model training
    """

    embedding_dict = defaultdict(lambda: [])
    net.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=8)
    with torch.no_grad():
        for b in tqdm(dataloader):
            img, indices = b
            features = net(img.cuda())
            indices = indices.numpy()
            features = features.cpu().numpy()
            for i, idx in enumerate(indices):
                file_name = dataloader.dataset.tiles[idx]
                bag_name, loc = file_name.strip('.jpeg').rsplit('/', 1)
                embedding_dict[bag_name].append((loc, features[i]))
            # break
    embedding_dict = {k: np.array(v, dtype=object) for k, v in embedding_dict.items()}
    return embedding_dict


def epoch_pseudolabel(bag_ins_list, criterion, milnet, args):
    """Used for evaluation. """
    bag_labels = []
    bag_predictions = []
    ins_predictions = {}
    ins_predictions_list = []
    epoch_loss = 0

    with torch.no_grad():
        for i, data in enumerate(bag_ins_list):
            # data  bag_label, ins feature, ins names, bag name
            bag_labels.append(np.clip(data[0], 0, 1))
            if not data[3] in ins_predictions.keys():
                ins_predictions[data[3]] = {}

            data_tensor = torch.from_numpy(np.stack(data[1])).float().cuda()
            data_tensor = data_tensor[:, 0:args.num_feats]
            label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().cuda()
            ins_pred, bag_prediction, _, _ = milnet(data_tensor)  # n X L
            ins_pred = torch.sigmoid(ins_pred)

            for ins_indx, ins_name in enumerate(data[2]):
                ins_predictions[data[3]][ins_name] = float(ins_pred[ins_indx].cpu().squeeze().numpy())
                #                 print(ins_predictions[data[3]][ins_name])
                ins_predictions_list.append(ins_predictions[data[3]][ins_name])

            loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            loss_total = loss_bag
            loss_total = loss_total.mean()
            bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())
            epoch_loss = epoch_loss + loss_total.item()
    epoch_loss = epoch_loss / len(bag_ins_list)
    return epoch_loss, bag_labels, bag_predictions, ins_predictions, ins_predictions_list


class SimCLR(object):
    """SimCLR training and evaluation pipeline."""

    def __init__(self, dataset, config, experiment, args):
        self.train_loader_pos, self.train_loader_neg, self.valid_loader, self.test_loader, self.train_loader_pos_full, self.train_loader_neg_clean = dataset
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.sup_nt_xent_criterion_pair2 = SupConLoss(temperature=config['loss']['temperature'],
                                                      pair_mode=2,
                                                      mask_uncertain_neg=config['mask_uncertain_neg'])
        self.sup_nt_xent_criterion_pair1 = SupConLoss(temperature=config['loss']['temperature'],
                                                      pair_mode=1,
                                                      mask_uncertain_neg=False)
        self.experiment = experiment
        self.args = args
        self.global_optimal_ac_val = 0
        self.update_signal = True

    def spl_scheduler(self, current_epoch):
        max_epoch = self.args.epochs
        warmup_epoch = self.args.warmup
        ro = self.args.ro
        ro_neg = self.args.ro_neg
        rT = self.args.rT
        return (current_epoch - warmup_epoch) * (rT - ro) / (max_epoch - warmup_epoch) + ro, (
                    current_epoch - warmup_epoch) * (rT - ro_neg) / (max_epoch - warmup_epoch) + ro_neg

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, labels, bag_label, n_iter, pair_mode=2):
        """Validation step"""
        # get the representations and the projections
        zis = model(xis)  # [N,C]

        # get the representations and the projections
        zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        features = torch.cat([zis.unsqueeze(1), zjs.unsqueeze(1)], dim=1)

        # loss = self.nt_xent_criterion(zis, zjs)

        if pair_mode == 2:
            loss = self.sup_nt_xent_criterion_pair2(features, labels, bag_label)
        else:
            loss = self.sup_nt_xent_criterion_pair1(features, labels, bag_label)

        return loss

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for xis, xjs, bag_label, label, gt_label, slide_name, patch_name in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, label, bag_label, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

    def _step_train(self, model, xis_pos, xjs_pos, xis_neg, xjs_neg, labels, bag_label, n_iter, pair_mode=2):
        """
        Training steps
        """

        # get the representations and the projections
        zis_pos = model(xis_pos)  # [N,C]

        # get the representations and the projections
        zjs_pos = model(xjs_pos)  # [N,C]

        # normalize projection feature vectors
        zis_pos = F.normalize(zis_pos, dim=1)
        zjs_pos = F.normalize(zjs_pos, dim=1)

        # get the representations and the projections
        zis_neg = model(xis_neg)  # [N,C]

        # get the representations and the projections
        zjs_neg = model(xjs_neg)  # [N,C]

        # normalize projection feature vectors
        zis_neg = F.normalize(zis_neg, dim=1)
        zjs_neg = F.normalize(zjs_neg, dim=1)

        zis = torch.cat([zis_neg, zis_pos])
        zjs = torch.cat([zjs_neg, zjs_pos])

        features = torch.cat([zis.unsqueeze(1), zjs.unsqueeze(1)], dim=1)

        if pair_mode == 2:
            loss = self.sup_nt_xent_criterion_pair2(features, labels, bag_label)
        else:
            loss = self.sup_nt_xent_criterion_pair1(features, labels, bag_label)

        return loss

    def train_MIL(self, root, epoch_count, args):
        """The function to train the MIL modules"""
        train_dataset = BagDataset(root + '/train_embeddings_epoch_0000.p')
        val_dataset = BagDataset(root + '/val_embeddings_epoch_0000.p')
        test_dataset = BagDataset(root + '/test_embeddings_epoch_0000.p')

        # this is the place to load the instance labels, should be the csv files
        # todo replace that with your own label path
        labels = pd.read_csv(self.args.labelroot + '/testing/reference.csv', header=None)

        labels_dict = {}
        for row_id in range(len(labels)):
            labels_dict[labels[0][row_id]] = labels[1][row_id]

        model_checkpoints_folder = root

        bags_list, val_list, test_list = [], [], []
        for b in train_dataset:
            bags_list.append([b[1].item(), b[0].numpy(), b[2], b[3]])
        for b in val_dataset:
            val_list.append([b[1].item(), b[0].numpy(), b[2], b[3]])

        for i, b in enumerate(test_dataset):
            test_list.append([int(labels.iloc[i][1] == 'Tumor')
                                 , b[0].numpy(), b[2], b[3]])

        i_classifier = mil.FCLayer(args.num_feats, 1)
        b_classifier = mil.BClassifier(input_size=args.num_feats, output_class=1)
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()

        pos_weight = torch.tensor(compute_pos_weight(bags_list))
        criterion = nn.BCEWithLogitsLoss(pos_weight)
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr_mil, weight_decay=args.weight_decay_mil)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 75, gamma=0.5, last_epoch=- 1, verbose=True)
        optimal_ac = 0
        auc_list = []
        auc_list_val = []
        optimal_ac_val = 0

        auc_list_val_optimal_test = []

        for epoch in range(0, args.num_epoch_mil):
            train_loss = epoch_train(bags_list, optimizer, criterion, milnet, args)  # iterate all bags

            # validation set
            val_loss, bag_labels_val, bag_predictions_val = epoch_test(val_list, criterion, milnet, args)
            accuracy_val, auc_value_val, precision_val, recall_val, fscore_val = five_scores(bag_labels_val,
                                                                                             bag_predictions_val)
            auc_list_val.append(auc_value_val)

            # test set
            test_loss, bag_labels, bag_predictions = epoch_test(test_list, criterion, milnet, args)
            accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_predictions)
            auc_list.append(auc_value)

            sys.stdout.write(
                '\r Epoch [%d/%d] train loss: %.4f, val loss: %.4f, test loss: %.4f, val auc: %.4f, accuracy: %.4f, auc score: %.4f, precision: %.4f, recall: %.4f, fscore: %.4f ' %
                (epoch + 1, args.num_epoch_mil, train_loss, val_loss, test_loss, auc_value_val, accuracy, auc_value,
                 precision, recall, fscore))

            if auc_value > optimal_ac:
                optimal_ac = max(auc_value, optimal_ac)
                torch.save(milnet.state_dict(), os.path.join(model_checkpoints_folder, 'ins_attention_model.pth'))
            scheduler.step()

            if fscore_val > optimal_ac_val:
                optimal_ac_val = max(fscore_val, optimal_ac_val)
                torch.save(milnet.state_dict(), os.path.join(model_checkpoints_folder, 'ins_attention_model_val.pth'))
                auc_list_val_optimal_test.append(auc_value)
            if epoch == args.epoch_to_extract_mil:
                torch.save(milnet.state_dict(),
                           os.path.join(model_checkpoints_folder, 'ins_attention_model_epoch_extract.pth'))
            scheduler.step()

        print('\n Optimal_MIL_auc: %.4f ' % (optimal_ac))
        print('\n Optimal_MIL_auc_val: %.4f ' % (optimal_ac_val))

        self.experiment.log_metric('Optimal_MIL_auc', optimal_ac, epoch=epoch_count)
        self.experiment.log_metric('Last_MIL_auc', auc_value, epoch=epoch_count)
        self.experiment.log_metric('Test_MIL_auc_sel_by_val', auc_list_val_optimal_test[-1], epoch=epoch_count)

        self.experiment.log_metric('Optimal_MIL_fscore_val', optimal_ac_val, epoch=epoch_count)

        if optimal_ac_val >= self.global_optimal_ac_val:
            self.global_optimal_ac_val = optimal_ac_val
            self.update_signal = True
        else:
            self.update_signal = False

        self.experiment.log_metric('global_optimal_ac_val', self.global_optimal_ac_val, epoch=epoch_count)

        state_dict = torch.load(os.path.join(model_checkpoints_folder, 'ins_attention_model_epoch_extract.pth'))
        milnet.load_state_dict(state_dict)

        test_loss, bag_labels, bag_predictions, ins_predictions, ins_predictions_list = epoch_pseudolabel(test_list,
                                                                                                          criterion,
                                                                                                          milnet, args)

        with open(model_checkpoints_folder + '/ins_pseudo_label_test.p', 'wb') as handle:
            pickle.dump(ins_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        auc_value, fscore, opt_thresh, max_f1 = self.eval_pseudo_label(root, ins_predictions, train=False)
        self.experiment.log_metric('Test_ins_auc', auc_value, epoch=epoch_count)
        self.experiment.log_metric('Test_opt_thresh', opt_thresh, epoch=epoch_count)
        self.experiment.log_metric('Test_ins_max_f1', max_f1, epoch=epoch_count)

        full_list = bags_list + val_list
        test_loss, bag_labels, bag_predictions, ins_predictions, ins_predictions_list = epoch_pseudolabel(full_list,
                                                                                                          criterion,
                                                                                                          milnet, args)

        with open(model_checkpoints_folder + '/ins_pseudo_label_train.p', 'wb') as handle:
            pickle.dump(ins_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # record the training metrics
        auc_value, fscore, opt_thresh, max_f1 = self.eval_pseudo_label(root, ins_predictions)
        self.experiment.log_metric('Pseudo_ins_auc', auc_value, epoch=epoch_count)
        self.experiment.log_metric('Pseudo_opt_thresh', opt_thresh, epoch=epoch_count)
        self.experiment.log_metric('Pseudo_ins_max_f1', max_f1, epoch=epoch_count)

        auc_value, fscore, opt_thresh, max_f1 = self.eval_pseudo_label_malbag(root, ins_predictions)
        self.experiment.log_metric('Pseudo_ins_auc_malbag', auc_value, epoch=epoch_count)
        self.experiment.log_metric('Pseudo_opt_thresh_malbag', opt_thresh, epoch=epoch_count)
        self.experiment.log_metric('Pseudo_ins_max_f1_malbag', max_f1, epoch=epoch_count)

        auc_value, fscore, opt_thresh, max_f1 = self.eval_pseudo_label_malbag(root,
                                                                              self.train_loader_pos.dataset.pseudo_label_EMA)
        self.experiment.log_metric('EMA_Pseudo_ins_auc_malbag', auc_value, epoch=epoch_count)
        self.experiment.log_metric('EMA_Pseudo_opt_thresh_malbag', opt_thresh, epoch=epoch_count)
        self.experiment.log_metric('EMA_Pseudo_ins_max_f1_malbag', max_f1, epoch=epoch_count)

    def eval_pseudo_label_malbag(self, root, ins_predictions, train=True):
        """Only evaluate the pseudo label for the malignant bags"""

        ins_gt_dir = self.args.labelroot + "/annotation/"
        if train:
            train_embedding = pickle.load(open(root + '/train_embeddings_epoch_0000.p', 'rb'))
            train_gt = pickle.load(open(ins_gt_dir + "gt_ins_labels_train.p", 'rb'))
        else:
            train_embedding = pickle.load(open(root + '/test_embeddings_epoch_0000.p', 'rb'))
            train_gt = pickle.load(open(ins_gt_dir + "gt_ins_labels_test.p", 'rb'))
        pred_test = ins_predictions

        train_name = np.stack([(k, row[0]) for k, v in train_embedding.items() for row in v])
        train_y_list = []
        y_prob_list = []
        for name in train_name:
            if "tumor" in name[0]:
                train_y_list.append(train_gt[name[0]][name[1]])
                y_prob_list.append(pred_test[name[0]][name[1]])
        train_y = np.array(train_y_list)
        y_prob = np.array(y_prob_list)
        accuracy, auc_value, precision, recall, fscore = five_scores(train_y, y_prob)
        precision, recall, thresholds = precision_recall_curve(train_y, y_prob)

        return auc_value, fscore, thresholds[
            np.argmax(2 * recall * precision / (recall + precision + 0.0000000001))], np.max(
            2 * recall * precision / (recall + precision + 0.0000000001))

    def eval_pseudo_label(self, root, ins_predictions, train=True):
        """Evaluate the pseudo label quality for the overall cases"""
        ins_gt_dir = self.args.labelroot + "/annotation/"
        if train:
            train_embedding = pickle.load(open(root + '/train_embeddings_epoch_0000.p', 'rb'))
            train_gt = pickle.load(open(ins_gt_dir + "gt_ins_labels_train.p", 'rb'))
        else:
            train_embedding = pickle.load(open(root + '/test_embeddings_epoch_0000.p', 'rb'))
            train_gt = pickle.load(open(ins_gt_dir + "gt_ins_labels_test.p", 'rb'))
        pred_test = ins_predictions
        train_name = np.stack([(k, row[0]) for k, v in train_embedding.items() for row in v])
        train_y = np.array([train_gt[name[0]][name[1]] for name in train_name])
        y_prob = np.array([pred_test[name[0]][name[1]] for name in train_name])
        accuracy, auc_value, precision, recall, fscore = five_scores(train_y, y_prob)
        precision, recall, thresholds = precision_recall_curve(train_y, y_prob)

        return auc_value, fscore, thresholds[
            np.argmax(2 * recall * precision / (recall + precision + 0.0000000001))], np.max(
            2 * recall * precision / (recall + precision + 0.0000000001))

    def extract_feature(self, state_dict_path, args):
        """The function to extract the feature representation."""
        model_dir = args.model_dir
        augmentation = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])
        }
        root_dir = '/single'

        train_dataset = SSLDataset(root_dir, 'dataset/train_bags.txt',
                                   transform=augmentation['val'], witness_rate=args.witness_rate, labelroot=args.labelroot)
        val_dataset = SSLDataset(root_dir, 'dataset/val_bags.txt',
                                 transform=augmentation['val'], witness_rate=args.witness_rate,  labelroot=args.labelroot)
        test_dataset = SSLDataset(root_dir, 'dataset/test_bags.txt',
                                  transform=augmentation['val'], witness_rate=args.witness_rate,  labelroot=args.labelroot)

        net = models.resnet18(num_classes=256, norm_layer=nn.InstanceNorm2d)
        net.fc = nn.Identity()
        net.cuda()
        net = nn.DataParallel(net)

        # load the model weight while neglect the fc layers
        state_dict_weights = torch.load(state_dict_path)['net']
        for i in range(2):
            state_dict_weights.popitem()

        state_dict_init = net.state_dict()
        new_state_dict = OrderedDict()

        print(state_dict_weights.keys())
        print(len(state_dict_weights.keys()))
        print(state_dict_init.keys())
        print(len(state_dict_init.keys()))
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

        for name, dl in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
            embedding_dict = get_features(net, dl)

            pickle.dump(embedding_dict, open(model_dir + "/{}_embeddings_epoch_{:04d}.p".format(name, 0), "wb"))

    def train(self):
        """The main training function."""
        model = ResNetSimCLR(**self.config["model"])  # .to(self.device)
        model = self._load_pre_trained_weights(model)
        if self.config['n_gpu'] > 1:
            device_n = len(eval(self.config['gpu_ids']))
            print(device_n)
            model = torch.nn.DataParallel(model, device_ids=range(device_n))
        model = model.to(self.device)

        # optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(self.config['weight_decay']))
        optimizer = torch.optim.SGD(model.parameters(), self.config['lr'],
                                    weight_decay=eval(self.config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.config['model_save_root'], self.config['expname'])

        if not os.path.exists(model_checkpoints_folder):
            os.mkdir(model_checkpoints_folder)

        # save config file
        _save_config_file(model_checkpoints_folder)
        saving_state = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(saving_state, os.path.join(model_checkpoints_folder, 'epoch' + 'init' + 'model.pth'))

        self.args.model_dir = os.path.join(model_checkpoints_folder, 'epoch_init')

        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        if self.args.init_MIL_training:
            # train the MIL modules using the inital extracted feature.
            self.extract_feature(os.path.join(model_checkpoints_folder, 'epoch' + 'init' + 'model.pth'),
                                 self.args)
            # train MIL and get the instance pseudo label
            self.train_MIL(self.args.model_dir, -1, self.args)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        # use for the neg-neg as positive pairs in the inital training
        for epoch_counter in range(self.config['epochs']):
            pos_dataloader_iterator = iter(self.train_loader_pos_full)
            pos_dataloader_iterator_filtered = iter(self.train_loader_pos)
            neg_dataloader_iterator_augmented = iter(self.train_loader_neg)

            for xis, xjs, bag_label, label, gt_label, slide_name, patch_name in self.train_loader_neg_clean:
                try:
                    xis_pos, xjs_pos, bag_label_pos, label_pos, gt_label_pos, slide_name_pos, patch_name_pos = next(
                        pos_dataloader_iterator)
                except StopIteration:
                    pos_dataloader_iterator = iter(self.train_loader_pos_full)
                    xis_pos, xjs_pos, bag_label_pos, label_pos, gt_label_pos, slide_name_pos, patch_name_pos = next(
                        pos_dataloader_iterator)

                xis_neg = xis
                xjs_neg = xjs
                xis_neg = xis_neg.to(self.device)
                xjs_neg = xjs_neg.to(self.device)

                xis_pos = xis_pos.to(self.device)
                xjs_pos = xjs_pos.to(self.device)

                bag_label = torch.cat([bag_label, bag_label_pos])
                label = torch.cat([label, label_pos])

                optimizer.zero_grad()

                label = label.to(self.device)

                # negneg
                loss_neg_sup = self._step_train(model, xis_pos, xjs_pos, xis_neg, xjs_neg, label, bag_label, n_iter,
                                                pair_mode=2)
                loss = loss_neg_sup

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    with self.experiment.train():
                        self.experiment.log_metric('train_loss', loss, step=n_iter)
                        self.experiment.log_metric('loss_neg_sup', loss_neg_sup, step=n_iter)
                        # self.experiment.log_metric('loss_glolocal', loss_glolocal, step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                # after the warmup epochs, add the positive instance as the query instance
                if epoch_counter >= self.args.warmup:
                    try:
                        xis_pos, xjs_pos, bag_label_pos, label_pos, gt_label_pos, slide_name_pos, patch_name_pos = next(
                            pos_dataloader_iterator_filtered)
                    except StopIteration:
                        pos_dataloader_iterator_filtered = iter(self.train_loader_pos)
                        xis_pos, xjs_pos, bag_label_pos, label_pos, gt_label_pos, slide_name_pos, patch_name_pos = next(
                            pos_dataloader_iterator_filtered)

                    try:
                        xis, xjs, bag_label, label, gt_label, slide_name, patch_name = next(
                            neg_dataloader_iterator_augmented)
                    except StopIteration:
                        neg_dataloader_iterator_augmented = iter(self.train_loader_neg)
                        xis, xjs, bag_label, label, gt_label, slide_name, patch_name = next(
                            neg_dataloader_iterator_augmented)

                    xis_neg = xis
                    xjs_neg = xjs
                    xis_neg = xis_neg.to(self.device)
                    xjs_neg = xjs_neg.to(self.device)

                    xis_pos = xis_pos.to(self.device)
                    xjs_pos = xjs_pos.to(self.device)

                    bag_label = torch.cat([bag_label, bag_label_pos])
                    label = torch.cat([label, label_pos])

                    optimizer.zero_grad()

                    label = label.to(self.device)
                    # pospos
                    loss_pos_sup = self._step_train(model, xis_pos, xjs_pos, xis_neg, xjs_neg, label, bag_label, n_iter,
                                                    pair_mode=1)
                    loss = loss_pos_sup

                    if n_iter % self.config['log_every_n_steps'] == 0:
                        self.writer.add_scalar('train_loss_pair1', loss, global_step=n_iter)
                        with self.experiment.train():
                            self.experiment.log_metric('loss_pos_sup_pair1', loss_pos_sup, step=n_iter)

                    if apex_support and self.config['fp16_precision']:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                n_iter += 1

            saving_state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            torch.save(saving_state, os.path.join(model_checkpoints_folder, 'epoch' + str(epoch_counter) + 'model.pth'))

            # begin self-paced learning to set the ratio of samples used in training
            if epoch_counter >= self.args.warmup:
                r, r_neg = self.spl_scheduler(epoch_counter)

                self.experiment.log_metric('spl_ratio', r, step=n_iter)
                self.train_loader_pos.dataset.ratio = r
                self.train_loader_pos.dataset.init_bag()
                self.experiment.log_metric('len(pos_bag)', len(self.train_loader_pos.dataset.tiles), step=n_iter)

                self.train_loader_neg.dataset.ratio = r_neg
                self.train_loader_neg.dataset.init_bag()
                self.experiment.log_metric('len(neg_bag)', len(self.train_loader_neg.dataset.tiles), step=n_iter)

                self.train_loader_neg_clean.dataset.ratio = r_neg
                self.train_loader_neg_clean.dataset.init_bag()
                self.experiment.log_metric('len(neg_bag_clean)', len(self.train_loader_neg_clean.dataset.tiles),
                                           step=n_iter)

            # updating the instance pseudo labels
            if epoch_counter >= 1 and epoch_counter % self.config['MIL_every_n_epochs'] == 0:
                self.args.model_dir = os.path.join(model_checkpoints_folder, 'epoch' + str(epoch_counter))

                if not os.path.exists(self.args.model_dir):
                    os.mkdir(self.args.model_dir)

                self.extract_feature(os.path.join(model_checkpoints_folder, 'epoch' + str(epoch_counter) + 'model.pth'),
                                     self.args)

                # train MIL and get the instance pseudo label
                self.train_MIL(self.args.model_dir, epoch_counter, self.args)
                pseudo_label_path = os.path.join(self.args.model_dir, 'ins_pseudo_label_train.p')

                if self.args.update_pseudo_label == True:
                    if self.update_signal:
                        self.train_loader_pos.dataset.update_pseudo_label(pseudo_label_path)
                        self.train_loader_pos.dataset.init_bag()

                        self.train_loader_neg.dataset.mask_uncertain_neg = False
                        self.train_loader_neg.dataset.update_pseudo_label(pseudo_label_path)
                        self.train_loader_neg.dataset.init_bag()

                        self.train_loader_pos_full.dataset.update_pseudo_label(pseudo_label_path)
                        self.train_loader_pos_full.dataset.init_bag()

                        self.train_loader_neg_clean.dataset.update_pseudo_label(pseudo_label_path)
                        self.train_loader_neg_clean.dataset.init_bag()

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, self.valid_loader)
                with self.experiment.test():
                    self.experiment.log_metric('val_loss', valid_loss, step=n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print('saved')

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        """load pretrain weights"""
        try:
            if self.config['fine_tune_from']:
                if '/ssl/moco/' in self.config['fine_tune_from']:
                    load_pretrained(model, self.config['fine_tune_from'])
                elif '/ssl/simclr' in self.config['fine_tune_from'] or '/ssl/simclr_new_split/' in self.config[
                    'fine_tune_from']:
                    load_pretrained_simCLR(model, self.config['fine_tune_from'])
                else:
                    load_pretrained_other(model, self.config['fine_tune_from'])
                    print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

def load_pretrained_simCLR(net, model_dir):
    print(model_dir)
    checkpoint = torch.load(model_dir)
    print(checkpoint.keys())
    print(checkpoint['state_dict'].keys())
    net_state_keys = list(net.state_dict().keys())
    print(net_state_keys[0])
    multigpu = False
    if 'module' in net_state_keys[0]:
        multigpu = True

    if not multigpu:
        model_state_dict = {k.replace("module.resnet.", "resnet."): v for k, v in
                            checkpoint['state_dict'].items() if
                            "resnet" in k}
    else:
        model_state_dict = {k.replace("module.resnet.", "module.resnet."): v for k, v in
                            checkpoint['state_dict'].items() if
                            "resnet" in k}

    print(model_state_dict.keys())
    print(net.state_dict().keys())
    net.load_state_dict(model_state_dict)
    print('load successfully')

def load_pretrained(net, model_dir):
    print(model_dir)
    checkpoint = torch.load(model_dir)
    net_state_keys = list(net.state_dict().keys())

    print(net_state_keys[0])

    multigpu = False
    if 'module' in net_state_keys[0]:
        multigpu = True

    if not multigpu:
        model_state_dict = {k.replace("module.encoder_q.", "resnet."): v for k, v in
                            checkpoint['state_dict'].items() if
                            "encoder_q" in k}
    else:
        model_state_dict = {k.replace("module.encoder_q.", "module.resnet."): v for k, v in
                            checkpoint['state_dict'].items() if
                            "encoder_q" in k}

    print(model_state_dict.keys())
    print(net.state_dict().keys())
    net.load_state_dict(model_state_dict)
    print('load successfully')

def load_pretrained_other(net, model_dir):
    print(model_dir)
    checkpoint = torch.load(model_dir)
    print(checkpoint['net'].keys())
    # print(checkpoint['state_dict'].keys())
    net_state_keys = list(net.state_dict().keys())

    print(net_state_keys[0])

    multigpu = False
    if 'module' in net_state_keys[0]:
        multigpu = True

    if not multigpu:
        model_state_dict = {k.replace("module.resnet.", "resnet."): v for k, v in
                            checkpoint['net'].items() if
                            "resnet" in k}
    else:
        model_state_dict = {k.replace("module.resnet.", "module.resnet."): v for k, v in
                            checkpoint['net'].items() if
                            "resnet" in k}

    print(model_state_dict.keys())
    print(net.state_dict().keys())
    net.load_state_dict(model_state_dict, strict=False)
    print('load successfully')

