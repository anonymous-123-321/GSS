import logging
from tqdm import tqdm
import torch
from collections import defaultdict
from .trian_util import check_best_loss, check_best_score, save_best_model
from .trian_util import add_logs_tensorboard,add_loss_details
import torch.nn.functional as F


def train(train_loader, val_loader, model, epochs, device, metric, log_dir, args):
    best_loss = None
    best_score = None
    best_hm_epoch = None
    best_loss_epoch = None

    visual_losses_train = defaultdict(list)
    visual_losses_val = defaultdict(list)
    visual_hm_epoch = []
    visual_lr_epochs=[]

    for epoch in range(epochs):
        
        visual_train_loss_epoch,visual_lr_epoch=train_step(train_loader, model,epoch,epochs, device, metric, args)
        
        _, val_hm,visual_val_loss_epoch = val_step(val_loader, model, epoch, epochs,  device, metric, args)
        
        best_score, best_hm_epoch = check_best_score(epoch, best_score, best_hm_epoch, val_hm, model,log_dir)
        
        model.optimize_scheduler(val_hm)

        for k ,v in visual_train_loss_epoch.items(): visual_losses_train[k].append(v)
        for k ,v in visual_val_loss_epoch.items(): visual_losses_val[k].append(v)
        visual_hm_epoch.append(val_hm)
        visual_lr_epochs.append(visual_lr_epoch)

    if args.best_model_criterion == 'loss': 
        return best_loss, best_score, best_loss_epoch
    elif args.best_model_criterion == 'score':
        return best_loss, best_score, best_hm_epoch, visual_losses_train, visual_losses_val,visual_hm_epoch,visual_lr_epochs


def train_step(data_loader, model, epoch,epochs, device, metric,  args):
    logger = logging.getLogger()
    model.train()
    metric.reset()

    targer2index=data_loader.dataset.target2index
    text_embeddings_all_current=data_loader.dataset.all_data['text'].to(device) #(v+a)

    batch_loss_details={}
    visual_loss_epoch=defaultdict(list)
    visual_lr_epoch=[]
    for batch_idx, batch in tqdm(enumerate(data_loader),mininterval=args.mininterval):

        audio_embeddings = batch["audio"].to(device)
        box_wav_embeddings = batch["box_wav"].to(device)

        video_embeddings=batch["video"].to(device)
        box_jpg_embeddings = batch["box_jpg"].to(device)

        text_embeddings=batch["text"].to(device)
        targets=batch["target"].to(device)

        
        targets2label=torch.tensor([targer2index[int(i)] for i in targets],device=device)

        loss_total, loss_details,visual_lr = model.optimize_params( #forward+loss
            audio_embeddings=audio_embeddings,
            video_embeddings=video_embeddings,
            targets=targets2label,
            text_embeddings=text_embeddings,
            text_embeddings_all_current=text_embeddings_all_current,
            args=args,
            optimize=True,
            box_jpg_embeddings=box_jpg_embeddings,
            box_wav_embeddings=box_wav_embeddings,
        )

        batch_loss_details=add_loss_details(loss_details, batch_loss_details)
        for k,v in loss_details.items(): visual_loss_epoch[k].append(v.item())
        visual_lr_epoch.append(visual_lr)

        iteration = len(data_loader) * epoch + batch_idx

    
    average_loss_details = {k: v / (batch_idx+1) for k, v in batch_loss_details.items()}
    loss_msg = " | ".join([f"{k}: {v.item():.4f}" for k, v in average_loss_details.items()])
    logger.info(
        f"TRAIN\t"
        f"Epoch: {epoch}/{epochs}\t"
        f"Iteration: {iteration}\t"
        f"{loss_msg}"
    )
    return visual_loss_epoch,visual_lr_epoch


def val_step(data_loader, model,  epoch, epochs, device, metric, args=None):

    logger = logging.getLogger()
    model.eval()

    metric.reset()

    targer2index = data_loader.dataset.target2index
    text_embeddings_all_current = data_loader.dataset.all_data['text'].to(device)  # (v+a)

    with torch.no_grad():

        batch_loss_details={}
        visual_loss_epoch = defaultdict(list)
        for batch_idx, batch in tqdm(enumerate(data_loader),mininterval=args.mininterval,desc=""):

            audio_embeddings = batch["audio"].to(device)
            video_embeddings = batch["video"].to(device)
            text_embeddings = batch["text"].to(device)
            targets = batch["target"].to(device)

            targets2label=torch.tensor([targer2index[int(i)] for i in targets],device=device)

            loss, loss_details,_ = model.optimize_params( 
                audio_embeddings=audio_embeddings,
                video_embeddings=video_embeddings,
                targets=targets2label,
                text_embeddings=text_embeddings,
                text_embeddings_all_current=text_embeddings_all_current,
                args=args,
                optimize=False)

            batch_loss_details = add_loss_details(loss_details, batch_loss_details)
            for k, v in loss_details.items(): visual_loss_epoch[k].append(v.item())
            iteration = len(data_loader) * epoch + batch_idx

        
        metric()
        values=metric.value()
        hm_score = values.get("both_hm", None)
        zsl_score = values.get("both_zsl", None)
        seen_score = values.get("both_seen", None)
        unseen_score = values.get("both_unseen", None)

        average_loss_details = {k: v / (batch_idx + 1) for k, v in batch_loss_details.items()}
        # loss_msg = " | ".join([f"{k}: {v.item():.4f}" for k, v in average_loss_details.items()])

        logger.info(
            f"VALID\t"
            f"Epoch: {epoch}/{epochs}\t"
            f"Iteration: {iteration}\t"
            f"ZSL: {zsl_score:.4f}\t"
            f"Seen: {seen_score:.4f}\t"
            f"Unseen: {unseen_score:.4f}\t"
            f"HM: {hm_score:.4f}\t"
            f"total_loss: {average_loss_details['total_loss']:.4f}"
        )
    return average_loss_details['total_loss'], hm_score,visual_loss_epoch
