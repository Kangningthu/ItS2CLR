

rsync -rltv --chmod=Dg+s,ug+rw,o-rwxs,Fug+rw,o-rwx,+X    /Users/liukangning/Downloads/ITS2CLR kl3141@log-1.hpc.nyu.edu:/scratch/kl3141


srun -t4:00:00 -c8 --account=cds --mem=240GB --nodes=1 --gres=gpu:2 --job-name "supsimCLR_bag" --cpus-per-task 8  --pty bash



srun -t48:00:00 -c8 --account=cds --mem=240GB --nodes=1 --gres=gpu:rtx8000:2 --job-name "supsimCLR_bag" --cpus-per-task 8  --pty bash

singularity exec --nv  --overlay /scratch/kl3141/singularity_env/overlay-7.5GB-300K.ext3:ro  \
                --overlay  /scratch/wz727/MIL/dataset/WSI/CAMELYON16/processed_tiles_5x.sqf \
                /scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
                /bin/bash

source /ext3/env.sh


cd  /scratch/kl3141/ITS2CLR/train

python run.py  \
--expname 0628_SPL_v2_supsimCLR2gpu_simCLR_ssl_pretrained_2_threshold_0.3_iterative_update_fre_5_posi_ratio_0.2_allow_pseudo_in_pair2 \
--threshold 0.3 \
--gputype 1 \
--augment_transform 0 \
--batch_size 512 \
--pretrain_weight /scratch/wz727/MIL/mil/checkpoints/ssl/simclr_new_split/checkpoint_0510.pth.tar \
--pseudo_label_path /scratch/wz727/MIL/mil/checkpoints/ssl/simclr_new_split/checkpoint_0510/ins_pseudo_label_train.p \
--MIL_every_n_epochs 5 \
--epoch_to_extract_mil 199 \
--posi_batch_ratio 0.2 \
--ro 0.2 \
--ro_neg 0.2 \
--rT 0.8 \
--warmup 15 \
--init_MIL_training yes \
--root_dir /single \
--comet_api_key uScZNAEkRHcYuBrEukI0I4fnL \
--labelroot  /scratch/wz727/MIL/dataset/WSI/CAMELYON16 \
--workspace kangning
