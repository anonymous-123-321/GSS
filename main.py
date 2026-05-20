import sys
from src.args import args_main
from eval.get_evaluation import get_evaluation
from dataset_all.dataset_util import *
from eval.metrics import MeanClassAccuracy
from model.my_zero_model3 import My_zero_shot_model3
from train.train import train
from src.utils import fix_seeds, setup_experiment, print_model_size,save_json_file,visuallize_loss_curve_from_json
import random


def run():
    args, eval_args = args_main()
    args.seed = random.randint(0, 3500) if args.seed is None else args.seed

    best_epoch = None
    randstate = f"({random.randint(0, 9999):04d})"

    if args.run == 'stage-1' or args.run == 'all':
        args.retrain_all = False
        path_stage_1, best_epoch = main(args,'stage1'+randstate)
        eval_args.load_path_stage_A = path_stage_1

    if args.run == 'stage-2' or args.run == 'all':
        args.retrain_all = True
        if best_epoch is not None and  args.epochs_stage2==-1:
            args.epochs_stage2 = best_epoch + 1
            print('best_epoch after stage1: ', best_epoch+1)
        path_stage_2, _ = main(args,'stage2'+randstate)
        eval_args.load_path_stage_B = path_stage_2

    if args.run == 'eval' or args.run == 'all':
        assert eval_args.load_path_stage_A != None
        assert eval_args.load_path_stage_B != None
        get_evaluation(eval_args)


def main(args,stage):

    fix_seeds(args.seed)
    logger, log_dir = setup_experiment(args, stage,"epoch", "loss", "hm")

    
    train_dataset, val_dataset=get_dataset(args)
    
    train_sampler, val_sampler = Sampler_dataset(train_dataset, val_dataset, logger,args)

    
    train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler, 
            num_workers=args.num_workers,
            pin_memory=True
        )

    val_loader_metric = data.DataLoader( 
        dataset=val_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader_loss = data.DataLoader( 
        dataset=val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    args.epochs = args.epochs_stage1 if 'stage1' in stage else args.epochs_stage2
    model = My_zero_shot_model3(args,)
    print_model_size(model, logger)
    model.to(args.device)

    metric=MeanClassAccuracy(model=model,dataset=val_dataset,dataloader=val_loader_metric, device=args.device,distance_fn=args.distance_fn,args=args)

    logger.info(model)



    best_loss, best_score, best_epoch,visual_loss_train,visual_loss_val,visual_hm_epoch,visual_lr_epochs = train(
        train_loader=train_loader,
        val_loader=val_loader_loss,
        model=model,
        epochs=args.epochs_stage1 if 'stage1' in stage  else args.epochs_stage2,
        device=args.device,
        metric=metric,
        log_dir=log_dir,
        args=args
    )


    # save_json_file(visual_loss_train,log_dir/"visual_loss_train.json")
    # save_json_file(visual_loss_val,log_dir/"visual_loss_val.json")
    # save_json_file(visual_hm_epoch,log_dir/"visual_hm_epoch.json")

    logger.info(f"FINISHED. Run is stored at {log_dir}")

    return log_dir, best_epoch,


if __name__ == '__main__':
    run()
