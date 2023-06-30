from trainer import SimCLR as Trainer
import yaml
from dataset.dataloader_simCLR import InssepDataset, InsDataset, InssepSPLDataset, SSLDataset, _get_simclr_pipeline_transform, _get_CE_pipeline_transform, _get_weak_pipeline_transform
import os, glob
import pandas as pd
import argparse
import comet_ml
from comet_ml import Experiment
from torch.utils.data import DataLoader


def str2bool(v):
    """Convert string to boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def spl_scheduler(current_epoch, warmup_epoch, max_epoch=100, ro=0.2, rT=0.8):
    """Scheduler for SPL"""
    return (current_epoch-warmup_epoch)*(rT-ro)/(max_epoch-warmup_epoch) + ro


def generate_csv(args):
    """Generate csv file for all patches"""
    path_temp = os.path.join('/single/training', '*', '*.jpeg')
    patch_path = glob.glob(path_temp)  # /class_name/bag_name/*.jpeg
    df = pd.DataFrame(patch_path)
    df.to_csv('all_patches.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', help='Dataset folder name')
    parser.add_argument('--labelroot', type=str, default=None, help='place to store the image-level labels and annotation')
    parser.add_argument('--expname', type=str, default='supsimCLR', help='Experiment name: used for comet and save model')
    parser.add_argument('--root_dir', type=str, default='/single/', help='the root directory for the input data')

    parser.add_argument('--batch_size', type=int, default=512, help='the batch size')

    parser.add_argument('--posi_batch_ratio', type=float, default=0.5, help='the postive number rate')
    parser.add_argument('--posi_query_ratio', type=float, default=0.05, help='the postive number rate')

    parser.add_argument('--ro', type=float, default=0.2, help='initial bag ratio')
    parser.add_argument('--ro_neg', type=float, default=0.2, help='initial bag ratio for negtive')

    parser.add_argument('--rT', type=float, default=0.8, help='final bag ratio')
    parser.add_argument('--warmup', type=int, default=10, help='warmup epoch before SPL')
    parser.add_argument('--init_MIL_training', type=str2bool, default=True, help='conduct MIL training in the begining')

    parser.add_argument('--use_ema', type=str2bool, default=False, help='use EMA as the pseudo label, we did not use it')

    parser.add_argument('--gputype', type=int, default=0, help='Gpu type, 0: single gpu, 1: 2 gpus')
    parser.add_argument('--pretrain_weight', type=str, default=None, help='pretrained weight of the SSL model')
    parser.add_argument('--pseudo_label_path', type=str, default=None, help='initial instance pseudo label path of SSL model')
    parser.add_argument('--threshold', type=float, default=0.7, help='threshold of binarizing the pseudo label')

    parser.add_argument('--temperature', type=float, default=0.07, help='temperature')
    parser.add_argument('--model_save_root', type=str, default='/scratch/kl3141/dsmil-wsi/supsimclr/savemodel',
                        help='model save root')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate of training the feature extractor')


    parser.add_argument('--mask_uncertain_neg', type=str2bool, default=False, help='whether to mask the instance with negative pseudo label whose bag label is postitive')

    parser.add_argument('--augment_transform', type=int, default=0, help='which type of data augmentation')
    parser.add_argument('--MIL_every_n_epochs', type=int, default=10, help='conduct MIL training every number of epoch')


    # whether to update the pseudo label or not
    parser.add_argument('--update_pseudo_label', type=str2bool, default=True, help='whether to update pseudo label')


    parser.add_argument('--witness_rate', type=float, default=None, help='witness_rate for the dataset')
    parser.add_argument('--epochs', type=int, default=100, help='epoch to train the sup simCLR')


    # MIL training
    parser.add_argument('--num_epoch_mil', type=int, default=350, help='number of epochs to train MIL')
    parser.add_argument('--num_feats', type=int, default=512, help='number of features to train MIL')
    parser.add_argument('--lr_mil', type=float, default=2e-4, help='lr to train MIL')
    parser.add_argument('--weight_decay_mil', type=float, default=1e-4, help='weight decay to train MIL')
    parser.add_argument('--epoch_to_extract_mil', type=int, default=349, help='epoch to extract the MIL feature')

    parser.add_argument('--loss_weight_bag', type=float, default=0.5, help='loss weight for bag')
    parser.add_argument('--loss_weight_ins', type=float, default=0.5, help='loss weight for instance')

    # comet API key
    parser.add_argument('--comet_api_key', type=str, default=None, help='comet api key')
    parser.add_argument('--workspace', type=str, default=None, help='comet project name')



    args = parser.parse_args()
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    config['fine_tune_from'] = args.pretrain_weight

    config['loss']['temperature'] = args.temperature
    config['expname'] = args.expname
    config['model_save_root'] = args.model_save_root
    config['lr'] = args.lr
    config['warmup'] = args.warmup
    config['batch_size'] = args.batch_size
    config['mask_uncertain_neg'] = args.mask_uncertain_neg

    config['MIL_every_n_epochs'] =args.MIL_every_n_epochs
    config['epochs'] = args.epochs
    config['posi_batch_ratio'] =args.posi_batch_ratio
    config['posi_query_ratio'] =args.posi_query_ratio

    # gpu_ids = eval(config['gpu_ids'])
    if args.gputype == 0:
        gpu_ids = [0]
        config['n_gpu'] = 1
        config['gpu_ids'] = '(0)'
    else:
        gpu_ids = [0, 1]
        config['n_gpu'] = 2
        config['gpu_ids'] = '(0,1)'

    print(gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    if args.augment_transform== 0:
        transform = _get_simclr_pipeline_transform()
        print('simCLR transform')
    elif args.augment_transform== 1:
        transform = _get_CE_pipeline_transform()
        print('CE transform')
    elif args.augment_transform == 2:
        transform = _get_weak_pipeline_transform()
        print('weak transform')



    train_dataset_pos = InssepSPLDataset(args.root_dir, 'dataset/train_bags.txt', transform, pseudo_label=args.pseudo_label_path, threshold=args.threshold, witness_rate=args.witness_rate, posbag=True, mask_uncertain_neg=args.mask_uncertain_neg, ratio=args.ro, use_ema=args.use_ema, labelroot=args.labelroot)
    train_dataset_neg = InssepSPLDataset(args.root_dir, 'dataset/train_bags.txt', transform, pseudo_label=args.pseudo_label_path, threshold=args.threshold, witness_rate=args.witness_rate, posbag=False, mask_uncertain_neg=args.mask_uncertain_neg, ratio=0, use_ema=args.use_ema, labelroot=args.labelroot)

    train_dataset_pos_full = InssepDataset(args.root_dir, 'dataset/train_bags.txt', transform, pseudo_label=args.pseudo_label_path, threshold=args.threshold, witness_rate=args.witness_rate, posbag=True, mask_uncertain_neg=args.mask_uncertain_neg,labelroot=args.labelroot)
    train_dataset_neg_clean = InssepSPLDataset(args.root_dir, 'dataset/train_bags.txt', transform, pseudo_label=args.pseudo_label_path, threshold=args.threshold, witness_rate=args.witness_rate, posbag=False, mask_uncertain_neg=args.mask_uncertain_neg, ratio=0, use_ema=args.use_ema, labelroot=args.labelroot)

    val_dataset = InsDataset(args.root_dir, 'dataset/val_bags.txt', transform, pseudo_label=args.pseudo_label_path, threshold=args.threshold, witness_rate=args.witness_rate, labelroot=args.labelroot)
    test_dataset = InsDataset(args.root_dir, 'dataset/test_bags.txt', transform, pseudo_label=None, threshold=args.threshold, witness_rate=args.witness_rate, labelroot=args.labelroot)

    # get the loader for traditional loading dataset
    train_loader_pos, train_loader_neg, valid_loader, test_loader = get_train_validation_data_loaders(train_dataset_pos,train_dataset_neg, val_dataset, test_dataset, config)

    # get the loader for SPL
    train_loader_pos_full, train_loader_neg_clean = get_train_validation_data_loaders_SPL(train_dataset_pos_full, train_dataset_neg_clean, config)

    experiment = Experiment(
        api_key=args.comet_api_key,
        project_name="mil",
        workspace=args.workspace,
    )

    experiment.set_name(args.expname)


    trainer = Trainer((train_loader_pos, train_loader_neg, valid_loader, test_loader, train_loader_pos_full, train_loader_neg_clean), config, experiment, args)

    trainer.train()



def get_train_validation_data_loaders(train_dataset_pos, train_dataset_neg,  val_dataset, test_dataset, config):


    train_loader_pos = DataLoader(train_dataset_pos, batch_size=int(config['batch_size']*config['posi_query_ratio']),
                              num_workers=config['dataset']['num_workers'], drop_last=True, shuffle=True)
    train_loader_neg = DataLoader(train_dataset_neg, batch_size=int(config['batch_size']*(1-config['posi_query_ratio'])),
                              num_workers=config['dataset']['num_workers'], drop_last=True, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                              num_workers=config['dataset']['num_workers'], drop_last=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                              num_workers=config['dataset']['num_workers'], drop_last=True, shuffle=False)
    return train_loader_pos, train_loader_neg, valid_loader, test_loader


def get_train_validation_data_loaders_SPL(train_dataset_pos_full, train_dataset_neg_clean, config):

    train_loader_pos_full = DataLoader(train_dataset_pos_full, batch_size=int(config['batch_size']*config['posi_batch_ratio']),
                              num_workers=config['dataset']['num_workers'], drop_last=True, shuffle=True)
    train_loader_neg_clean = DataLoader(train_dataset_neg_clean, batch_size=int(config['batch_size']*(1-config['posi_batch_ratio'])),
                              num_workers=config['dataset']['num_workers'], drop_last=True, shuffle=True)

    return train_loader_pos_full, train_loader_neg_clean
if __name__ == "__main__":
    main()