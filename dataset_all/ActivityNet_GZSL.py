import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils import data
from src.utils import  get_class_names

class ActivityNetDataset(data.Dataset):
    def __init__(self, args, dataset_split,):
        super(ActivityNetDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset: {self.__class__.__name__}\t"
            f"split: {dataset_split}\t")
        self.args = args
        self.root = args.data_dir
        self.zero_shot_split = args.zero_shot_split
        self.dataset_split = dataset_split

        self.check_exist()
        self.get_class_names_and_idxs()
        self.get_current_seen_and_unseen_class_names_and_ids()
        self.data_pt = self.get_data_from_pt()  
        self.text_pt = torch.load("/mnt/sdc/your_name/Dataset/ActivityNet/GZSL_data/text_embedding.pt",
                                  map_location='cpu')
        self.all_data = self.all_data()
        self.target2index = self.map_target()
        self.target_to_indices = {target: np.where(self.all_data['target'] == target)[0]  
                                  for target in set(self.all_data['target'].tolist())}

    def __getitem__(self, item):
        audio_feature = self.all_data['audio'][item]
        video_feature = self.all_data['video'][item]

        target = self.all_data['target'][item]  
        text_index = self.target2index[target.item()]
        text_feature = self.all_data['text'][text_index]

        if "train" in self.dataset_split:
            box_jpg = self.all_data['box_jpg'][item]
            box_wav = self.all_data['box_wav'][item]
            return {
                'audio': audio_feature,
                'video': video_feature,
                'text': text_feature,
                'target': target,
                'box_jpg': box_jpg,
                'box_wav': box_wav
            }
        else:
            return {
                'audio': audio_feature,
                'video': video_feature,
                'text': text_feature,
                'target': target,
            }

    def __len__(self):
        return len(self.all_data['target'])

    def all_data(self): 
        classes_mask = np.where(np.isin(self.data_pt["audio"]["target"], self.current_seen_unseen_class_ids))[0] 
        return {
            "audio": torch.stack(self.data_pt["audio"]["data"])[classes_mask],
            "video": torch.stack(self.data_pt["video"]["data"])[classes_mask],
            "box_jpg": torch.stack(self.data_pt["video"]["box"])[
                classes_mask] if "train" in self.dataset_split else None,
            "box_wav": torch.stack(self.data_pt['audio']['box'])[
                classes_mask] if "train" in self.dataset_split else None,
            "text": torch.stack(self.text_pt["text"]["data"])[sorted(self.current_seen_unseen_class_ids.astype(int))],
            "target": self.data_pt["audio"]["target"][classes_mask],
        }

    def map_target(self):
        current_target = sorted(self.current_seen_unseen_class_ids)  # [5,12,13,...]
        target2index = {}
        for i in range(len(current_target)):  
            target2index[int(current_target[i])] = i
        return  target2index


    def get_class_names_and_idxs(self): 
        
        self.all_class_names=get_class_names(self.root / "class-split/all_class.txt")
        self.all_class_name_map_idx = {_class: i for i, _class in enumerate(sorted(self.all_class_names))}
        self.all_class_idx = np.asarray([self.all_class_name_map_idx[name] for name in self.all_class_names])

        
        self.train_train_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_train.txt")
        self.train_train_ids = np.asarray([self.all_class_name_map_idx[name] for name in self.train_train_class_names])

        
        self.val_seen_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")
        self.val_seen_ids = np.asarray([self.all_class_name_map_idx[name] for name in self.val_seen_class_names])
        self.val_unseen_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")
        self.val_unseen_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.val_unseen_class_names])

        
        self.test_train_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_train.txt")
        self.test_seen_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")
        self.test_unseen_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

        self.test_train_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.test_train_class_names])
        self.test_seen_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.test_seen_class_names])
        self.test_unseen_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.test_unseen_class_names])


    def get_current_seen_and_unseen_class_names_and_ids(self):
        if self.dataset_split == "train":
            self.current_seen_class_names=self.train_train_class_names
            self.current_unseen_class_names=np.array([])
        elif self.dataset_split == "val":
            self.current_seen_class_names = self.val_seen_class_names
            self.current_unseen_class_names = self.val_unseen_class_names
        elif self.dataset_split == "train_val":
            self.current_seen_class_names = np.concatenate((self.train_train_class_names, self.val_unseen_class_names))
            self.current_unseen_class_names = np.array([])
        elif self.dataset_split == "test":
            self.current_seen_class_names = self.test_seen_class_names
            self.current_unseen_class_names = self.test_unseen_class_names

        self.current_seen_class_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.current_seen_class_names])
        self.current_unseen_class_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.current_unseen_class_names])
        self.current_seen_unseen_class_ids=np.sort(np.concatenate((self.current_seen_class_ids, self.current_unseen_class_ids)))

    def get_data_from_pt(self):
        if self.dataset_split == "train":
            data_file = self.train_file_path
        elif self.dataset_split == "val":
            data_file = self.val_file_path
        elif self.dataset_split == "train_val":
            data_file = self.trainval_file_path
        elif self.dataset_split == "test":
            data_file = self.test_file_path

        self.logger.info(f"Loading processed data from {data_file}")
        x = torch.load(data_file, map_location='cpu')
        return x
    def check_exist(self):
        self.train_file_path = Path("/mnt/sdc/your_name/Dataset/ActivityNet/GZSL_data/stage_1_train.pt")
        self.trainval_file_path = Path("/mnt/sdc/your_name/Dataset/ActivityNet/GZSL_data/stage_2_train.pt")
        self.val_file_path = Path("/mnt/sdc/your_name/Dataset/ActivityNet/GZSL_data/stage_1_val.pt")
        self.test_file_path = Path("/mnt/sdc/your_name/Dataset/ActivityNet/GZSL_data/stage_2_test.pt")
        if self.train_file_path.exists() and self.trainval_file_path.exists() and self.val_file_path.exists() and self.test_file_path.exists():
            return True
        else:
            raise FileNotFoundError
