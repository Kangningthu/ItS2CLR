import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from collections import defaultdict
import torchvision.transforms as transforms
import pickle


class WSIDataset(Dataset):
    def __init__(self, root_dir, bag_names_file, transform, witness_rate=None, labelroot=None):
        self.transform = transform
        self.root_dir = root_dir
        self.bag_names = self._load_bag_name(bag_names_file)
        # self.bag2tiles = {b: os.listdir(root_dir + '/' + b)
        #                   for b in self.bag_names}
        self.bag2tiles = {b: os.listdir(root_dir + b)
                          for b in self.bag_names}
        self.labelroot = labelroot
        self.ins_gt = self._load_ins_gt()
        self.witness_rate = witness_rate
        if witness_rate:
            self._sample_bag(witness_rate)
        self._compute_avg_wr()


    def _load_ins_gt(self):
        ins_gt_dir = self.labelroot + "/annotation/"
        train_gt = pickle.load(open(ins_gt_dir + "gt_ins_labels_train.p", 'rb'))
        test_gt = pickle.load(open(ins_gt_dir + "gt_ins_labels_test.p", 'rb'))
        ins_gt = {**train_gt, **test_gt}
        return ins_gt

    def _sample_bag(self, witness_rate):
        np.random.seed(2021)
        self.bag_ratio = {}
        for k, v in self.bag2tiles.items():
            tile_gts = np.array([self.ins_gt["/" + k.strip("/")][tile.strip(".jpeg")] for tile in v]).astype('bool')
            pos_tiles = tile_gts.sum()
            neg_tiles = len(tile_gts) - pos_tiles
            self.bag_ratio[k] = [neg_tiles, pos_tiles]
            tile_list = np.array(v)
            if witness_rate < 1:
                if pos_tiles > 0:
                    self.bag2tiles[k] = np.concatenate(
                        [np.random.choice(tile_list[~tile_gts], int(neg_tiles * witness_rate), replace=False),
                         tile_list[tile_gts]])
            else:
                if pos_tiles > 0:
                    self.bag2tiles[k] = np.concatenate(
                        [np.random.choice(tile_list[tile_gts], int(pos_tiles / witness_rate), replace=False),
                         tile_list[~tile_gts]])

    def _compute_avg_wr(self):
        self._count_ratio()
        s = 0
        count = 0
        for k, (neg, pos) in self.bag_ratio.items():
            if pos > 0:
                s += pos / (neg + pos)
                count += 1
        self.avg_wr = s / count
        print("Avg. WR: {:.4f}".format(self.avg_wr))

    def _count_ratio(self):
        self.bag_ratio = {}
        for k, v in self.bag2tiles.items():
            tile_gts = np.array([self.ins_gt["/" + k.strip("/")][tile.strip(".jpeg")] for tile in v]).astype('bool')
            pos_tiles = tile_gts.sum()
            neg_tiles = len(tile_gts) - pos_tiles
            self.bag_ratio[k] = [neg_tiles, pos_tiles]

    def _load_bag_name(self, name_file):
        with open(name_file, 'r') as f:
            bag_list = f.read().split('\n')
        return bag_list

    def _get_data(self, src):
        image = cv2.imread(src)
        image = Image.fromarray(image)
        return image


class SSLDataset(WSIDataset):
    def __init__(self, root_dir, bag_names_file, transform, witness_rate=None,labelroot=None):
        super().__init__(root_dir, bag_names_file, transform, witness_rate,labelroot)
        self.tiles = sum([[k + v for v in vs] for k, vs in self.bag2tiles.items()], [])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        # print(self.root_dir)
        # print(self.tiles[index])
        # tile_dir = os.path.join(self.root_dir, self.tiles[index])
        tile_dir = self.root_dir + self.tiles[index]
        # print(tile_dir)
        img = self._get_data(tile_dir)
        img = self.transform(img)
        return img, index


# used for the instance evaluation
class InsDataset(WSIDataset):
    def __init__(self, root_dir, bag_names_file, transform, pseudo_label=None, threshold=0.7, witness_rate=None,
                 labelroot=None):
        super().__init__(root_dir, bag_names_file, transform, witness_rate, labelroot)
        self.tiles = sum([[k + v for v in vs] for k, vs in self.bag2tiles.items()], [])

        if 'train' in bag_names_file or 'val' in bag_names_file:
            gt_path = labelroot + '/annotation/gt_ins_labels_train.p'
            self.train_stat = True
        else:
            gt_path = labelroot + '/annotation/gt_ins_labels_test.p'
            self.train_stat = False
        self.gt_label = pickle.load(open(gt_path, 'rb'))

        if pseudo_label:
            self.pseudo_label = pickle.load(open(pseudo_label, 'rb'))
        else:
            self.pseudo_label = None
        self.threshold = threshold

    def update_pseudo_label(self, label_path):
        self.pseudo_label = pickle.load(open(label_path, 'rb'))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        # print(self.root_dir)
        # print(self.tiles[index])
        # tile_dir = os.path.join(self.root_dir, self.tiles[index])
        tile_dir = self.root_dir + self.tiles[index]
        # print(tile_dir)
        img = self._get_data(tile_dir)
        img = transforms.functional.to_tensor(img)
        # two augmentations

        img1 = self.transform(img)
        if self.train_stat:
            img2 = self.transform(img)
        # return img, index

        temp_path = tile_dir
        # print(temp_path)

        bag_label = 1 if "tumor" in temp_path else 0

        label = bag_label

        #        /single/training/normal_001/24_181.jpeg
        file_list = temp_path.split('.')[0]
        file_list = file_list.split('/')[-2:]

        if 'train' in tile_dir:
            slide_name = "/training/" + file_list[0]  # normal_001
        else:
            slide_name = "/testing/" + file_list[0]  # test_001
        patch_name = file_list[1]  # 24_181

        if self.pseudo_label:

            pseudo_label = self.pseudo_label[slide_name][patch_name]
            # print(pseudo_label)
            if pseudo_label > self.threshold and label == 1:
                label = 1
            else:
                label = 0

        if self.train_stat:
            return img1, img2, torch.LongTensor(np.array([bag_label])), torch.LongTensor(
                np.array([label])), torch.LongTensor([self.gt_label[slide_name][patch_name]]), slide_name, patch_name
        else:
            return img1, torch.LongTensor(np.array([bag_label])), torch.LongTensor(np.array([label])), torch.LongTensor(
                [self.gt_label[slide_name][patch_name]]), slide_name, patch_name


# separte the postive bag and the negative bag:
class InssepDataset(WSIDataset):
    def __init__(self, root_dir, bag_names_file, transform, pseudo_label=None, threshold=0.7, witness_rate=None,
                 posbag=True, mask_uncertain_neg=True, labelroot=None):
        super().__init__(root_dir, bag_names_file, transform, witness_rate, labelroot)
        self.threshold = threshold
        self.posbag = posbag
        self.mask_uncertain_neg = mask_uncertain_neg

        if pseudo_label:
            self.pseudo_label = pickle.load(open(pseudo_label, 'rb'))
        else:
            self.pseudo_label = None

        self.init_bag()

        # self.tiles = sum([[k + v for v in vs] for k, vs in self.bag2tiles.items()], [])

        if 'train' in bag_names_file or 'val' in bag_names_file:
            gt_path = labelroot + '/annotation/gt_ins_labels_train.p'
            self.train_stat = True
            # if 'train' in bag_names_file:
            #     self.train_stat = True
            # else:
            #     # validation
            #     self.train_stat = False
        # test
        else:
            gt_path = labelroot + '/annotation/gt_ins_labels_test.p'
            self.train_stat = False
        self.gt_label = pickle.load(open(gt_path, 'rb'))

    def init_bag(self):
        self.tiles = []
        if self.posbag == True:
            for k, vs in self.bag2tiles.items():
                if "tumor" in k:
                    for v in vs:
                        if self.pseudo_label[k[:-1]][v.split('.')[0]] > self.threshold:
                            self.tiles.append(k + v)
        else:
            for k, vs in self.bag2tiles.items():
                if not self.mask_uncertain_neg:
                    if "tumor" in k:
                        for v in vs:
                            if self.pseudo_label[k[:-1]][v.split('.')[0]] <= self.threshold:
                                self.tiles.append(k + v)

                if not "tumor" in k:
                    for v in vs:
                        self.tiles.append(k + v)

        # print( self.tiles)
        print('self.posbag')
        print(self.posbag)
        print(len(self.tiles))

    def update_pseudo_label(self, label_path):
        self.pseudo_label = pickle.load(open(label_path, 'rb'))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile_dir = self.root_dir + self.tiles[index]
        # print(tile_dir)
        img = self._get_data(tile_dir)
        img = transforms.functional.to_tensor(img)
        # two augmentations

        img1 = self.transform(img)
        if self.train_stat:
            img2 = self.transform(img)
        # return img, index

        temp_path = tile_dir
        # print(temp_path)

        bag_label = 1 if "tumor" in temp_path else 0

        label = bag_label

        #        /single/training/normal_001/24_181.jpeg
        file_list = temp_path.split('.')[0]  # /single/training/normal_001/24_18
        file_list = file_list.split('/')[-2:]  # [normal_001,24_18]

        if 'train' in tile_dir:
            slide_name = "/training/" + file_list[0]  # normal_001
        else:
            slide_name = "/testing/" + file_list[0]  # test_001
        patch_name = file_list[1]  # 24_181

        if self.pseudo_label:

            pseudo_label = self.pseudo_label[slide_name][patch_name]
            # print(pseudo_label)
            if pseudo_label > self.threshold and label == 1:
                label = 1
            else:
                label = 0

        if self.train_stat:
            return img1, img2, torch.LongTensor(np.array([bag_label])), torch.LongTensor(
                np.array([label])), torch.LongTensor([self.gt_label[slide_name][patch_name]]), slide_name, patch_name
        else:
            return img1, torch.LongTensor(np.array([bag_label])), torch.LongTensor(np.array([label])), torch.LongTensor(
                [self.gt_label[slide_name][patch_name]]), slide_name, patch_name


# separte the postive bag and the negative bag, add the SPL support
class InssepSPLDataset(WSIDataset):
    def __init__(self, root_dir, bag_names_file, transform, pseudo_label=None, threshold=0.7, witness_rate=None,
                 posbag=True, mask_uncertain_neg=True, ratio=0.8, use_ema=False, labelroot=None):
        super().__init__(root_dir, bag_names_file, transform, witness_rate,labelroot)
        self.threshold = threshold
        self.posbag = posbag
        self.mask_uncertain_neg = mask_uncertain_neg
        self.use_ema = use_ema

        if pseudo_label:
            self.pseudo_label = pickle.load(open(pseudo_label, 'rb'))
            self.pseudo_label_EMA = self.pseudo_label
        else:
            self.pseudo_label = None
            self.pseudo_label_EMA = None

        self.ratio = ratio
        self.init_bag()

        if 'train' in bag_names_file or 'val' in bag_names_file:
            gt_path = labelroot + '/annotation/gt_ins_labels_train.p'
            self.train_stat = True
            # if 'train' in bag_names_file:
            #     self.train_stat = True
            # else:
            #     # validation
            #     self.train_stat = False
        # test
        else:
            gt_path = labelroot + '/annotation/gt_ins_labels_test.p'
            self.train_stat = False
        self.gt_label = pickle.load(open(gt_path, 'rb'))

    def init_bag(self):
        self.tiles = []
        self.instance_confidence = []
        if self.posbag == True:
            for k, vs in self.bag2tiles.items():
                if "tumor" in k:
                    for v in vs:
                        # print(k[:-1])
                        # print(v.split('.')[0])
                        # print(self.pseudo_label[k[:-1]][v.split('.')[0]])
                        # print(self.threshold)

                        if self.pseudo_label[k[:-1]][v.split('.')[0]] > self.threshold:
                            # self.tiles.append(k + v)
                            self.instance_confidence.append(self.pseudo_label[k[:-1]][v.split('.')[0]])
            # print(self.instance_confidence)
            print(1 - self.ratio)
            threshold_new = np.quantile(self.instance_confidence, 1 - self.ratio)
            print(threshold_new)

            for k, vs in self.bag2tiles.items():
                if "tumor" in k:
                    for v in vs:
                        # print(k[:-1])
                        # print(v.split('.')[0])
                        # print(self.pseudo_label[k[:-1]][v.split('.')[0]])
                        # print(self.threshold)

                        if self.pseudo_label[k[:-1]][v.split('.')[0]] >= threshold_new:
                            self.tiles.append(k + v)




        else:
            for k, vs in self.bag2tiles.items():
                if not self.mask_uncertain_neg:
                    if "tumor" in k:
                        for v in vs:
                            if self.pseudo_label[k[:-1]][v.split('.')[0]] <= self.threshold:
                                # self.tiles.append(k + v)
                                self.instance_confidence.append(self.pseudo_label[k[:-1]][v.split('.')[0]])

                if not "tumor" in k:
                    for v in vs:
                        self.tiles.append(k + v)

            if not len(self.instance_confidence) == 0:
                if not self.mask_uncertain_neg:
                    threshold_new = np.quantile(self.instance_confidence, self.ratio)
                    for k, vs in self.bag2tiles.items():
                        if "tumor" in k:
                            for v in vs:
                                if self.pseudo_label[k[:-1]][v.split('.')[0]] <= threshold_new:
                                    self.tiles.append(k + v)
                                    # self.instance_confidence.append(self.pseudo_label[k[:-1]][v.split('.')[0]])

        # print( self.tiles)
        print('self.posbag')
        print(self.posbag)
        print(len(self.tiles))

    def update_pseudo_label(self, label_path):
        pseudo_label = pickle.load(open(label_path, 'rb'))

        for slide_name in self.pseudo_label.keys():
            for patch_name in self.pseudo_label[slide_name].keys():
                self.pseudo_label_EMA[slide_name][patch_name] = 0.3 * self.pseudo_label_EMA[slide_name][patch_name] \
                                                                + 0.7 * pseudo_label[slide_name][patch_name]
        if self.use_ema:
            self.pseudo_label = self.pseudo_label_EMA
        else:
            self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        # print(self.root_dir)
        # print(self.tiles[index])
        # tile_dir = os.path.join(self.root_dir, self.tiles[index])
        tile_dir = self.root_dir + self.tiles[index]
        # print(tile_dir)
        img = self._get_data(tile_dir)
        img = transforms.functional.to_tensor(img)
        # two augmentations

        img1 = self.transform(img)
        if self.train_stat:
            img2 = self.transform(img)
        # return img, index

        temp_path = tile_dir
        # print(temp_path)

        bag_label = 1 if "tumor" in temp_path else 0

        label = bag_label

        #        /single/training/normal_001/24_181.jpeg
        file_list = temp_path.split('.')[0]  # /single/training/normal_001/24_18
        file_list = file_list.split('/')[-2:]  # [normal_001,24_18]

        if 'train' in tile_dir:
            slide_name = "/training/" + file_list[0]  # normal_001
        else:
            slide_name = "/testing/" + file_list[0]  # test_001
        patch_name = file_list[1]  # 24_181

        if self.pseudo_label:
            pseudo_label = self.pseudo_label[slide_name][patch_name]
            # print(pseudo_label)
            if pseudo_label > self.threshold and label == 1:
                label = 1
            else:
                label = 0

        if self.train_stat:
            return img1, img2, torch.LongTensor(np.array([bag_label])), torch.LongTensor(
                np.array([label])), torch.LongTensor([self.gt_label[slide_name][patch_name]]), slide_name, patch_name
        else:
            return img1, torch.LongTensor(np.array([bag_label])), torch.LongTensor(np.array([label])), torch.LongTensor(
                [self.gt_label[slide_name][patch_name]]), slide_name, patch_name


# the following two functions are used in the MIL training
class BagDataset(Dataset):
    def __init__(self, file_dir):
        self.embedding_dict = pickle.load(open(file_dir, 'rb'))
        self.slide_name = list(self.embedding_dict.keys())

    def __len__(self):
        return len(self.embedding_dict)

    def __getitem__(self, index):
        key = self.slide_name[index]
        feats = np.array([v[1] for v in self.embedding_dict[key]])
        label = 1 if "tumor" in key else 0
        return torch.FloatTensor(feats), torch.LongTensor(np.array([label]))


class BagDataset_ins(Dataset):
    def __init__(self, file_dir):
        self.embedding_dict = pickle.load(open(file_dir, 'rb'))
        self.slide_name = list(self.embedding_dict.keys())

    def __len__(self):
        return len(self.embedding_dict)

    def __getitem__(self, index):
        key = self.slide_name[index]
        feats = np.array([v[1] for v in self.embedding_dict[key]])
        ins_name = [v[0] for v in self.embedding_dict[key]]
        label = 1 if "tumor" in key else 0
        bag_name = key
        return torch.FloatTensor(feats), torch.LongTensor(np.array([label])), ins_name, bag_name


# data augmentation related function
class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        np.random.seed(0)

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            #            print(self.kernel_size)
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img


def _get_simclr_pipeline_transform():
    s = 1
    input_shape = (224, 224, 3)
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([ToPIL(),
                                          transforms.RandomResizedCrop(size=input_shape[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.06 * input_shape[0])),
                                          transforms.ToTensor()])
    return data_transforms


def _get_CE_pipeline_transform():
    s = 1
    input_shape = (224, 224, 3)
    # get a set of data augmentation transformations as described in the SimCLR paper.
    # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([ToPIL(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.06 * input_shape[0])),
                                          transforms.ToTensor()])

    return data_transforms


def _get_weak_pipeline_transform():
    s = 1
    input_shape = (224, 224, 3)
    # get a set of data augmentation transformations as described in the SimCLR paper.
    # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([ToPIL(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    return data_transforms
