#!/bin/bash




for i in {1..1}
do
  python main.py --epochs_stage1 15  --bs 64 --n_batches 300 --lr 0.0001 --mininterval 600  --lr_scheduler reduce \
  --exp_name "tiaocan_${i}"  --num_workers 2 \
  --dataset_name ActivityNet --data_dir "/mnt/sdc/your_name/Dataset/ActivityNet/" \
  --dropout 0.1 --mapper_hidden_size 512  --enc_hidden_size 512
  --margin_ratio 2.0
done


