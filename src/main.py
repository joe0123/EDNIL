import argparse
import os
import json
import torch
import torch.optim as optim
import numpy as np
import random

from datareaders import ColorMNIST
from models import MLP
from EnvInfer import EITrainer
from InvLearn import ILTrainer

def seed_torch(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--dataset", type=str, choices=["cmnist"])
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--joint_iters", type=int)
    parser.add_argument("--ei__num_envs", type=int)
    parser.add_argument("--ei__hidden_dims", nargs='*', type=int)
    parser.add_argument("--ei__pre_batch_size", type=int)
    parser.add_argument("--ei__batch_size", type=int)
    parser.add_argument("--ei__pre_lr", type=float)
    parser.add_argument("--ei__enc_lr", type=float)
    parser.add_argument("--ei__clf_lr", type=float)
    parser.add_argument("--ei__pre_l2_w", type=float)
    parser.add_argument("--ei__l2_w", type=float)
    parser.add_argument("--ei__pre_iters", type=int)
    parser.add_argument("--ei__iters", type=int)
    parser.add_argument("--ei__temperature", type=float)
    parser.add_argument("--ei__beta", type=float)
    parser.add_argument("--ei__gamma", type=float)
    parser.add_argument("--ei__envw_thres", type=float)
    parser.add_argument("--ei__pre_grad_clip_norm", type=float)
    parser.add_argument("--ei__grad_clip_norm", type=float)
    parser.add_argument("--ei__pre_log_times", type=int)
    parser.add_argument("--ei__log_times", type=int)
    parser.add_argument("--il__hidden_dims", nargs='*', type=int)
    parser.add_argument("--il__batch_size", type=int)
    parser.add_argument("--il__lr", type=float)
    parser.add_argument("--il__l2_w", type=float)
    parser.add_argument("--il__grad_clip_norm", type=float)
    parser.add_argument("--il__anneal_penalty_w", type=float)
    parser.add_argument("--il__penalty_w", type=float)
    parser.add_argument("--il__anneal_joint_iters", type=int)
    parser.add_argument("--il__iters", type=int)
    parser.add_argument("--il__log_times", type=int)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--runs", nargs='*', type=int)
    args = parser.parse_args()
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            vars(args).update(json.load(f))
    print(args)

    all_test_scores = []
    all_test_avgs = []
    for ri, rs in enumerate(args.runs, 1):
        seed_torch(rs)
        print("\n========================= Run {} =========================".format(ri))
        
        if args.dataset == "cmnist":
            train_envs, valid_envs, test_envs = ColorMNIST(data_root="../data", 
                                                           val_ratio=args.val_ratio)
            num_classes = 2   # Binary classification
            test_metric = "acc"
            ei_model = MLP(feat_dim=train_envs[0]['X'].shape[1], 
                           out_dim=args.ei__num_envs, 
                           hidden_dims=args.ei__hidden_dims)
            ei_pre_optimizer = optim.Adam(ei_model.parameters(), lr=args.ei__pre_lr)
            ei__pre_l2_w = args.ei__pre_l2_w
            opt_params = [{"params": ei_model.enc_params, "lr": args.ei__enc_lr}, 
                          {"params": ei_model.clf_params, "lr": args.ei__clf_lr}]
            ei_optimizer = optim.Adam(opt_params)
            ei__l2_w = args.ei__l2_w
            il_model = MLP(feat_dim=train_envs[0]['X'].shape[1], 
                           out_dim=1, 
                           hidden_dims=args.il__hidden_dims)
            il_optimizer = optim.Adam(il_model.parameters(), lr=args.il__lr)
            il__l2_w = args.il__l2_w
        else:
            raise NotImplementedError

        print("\nData stats:")
        for ei, train_env in enumerate(train_envs):
            print("Size of train env {}: {}".format(ei, len(train_env['y'])))
        for ei, valid_env in enumerate(valid_envs):
            print("Size of valid env {}: {}".format(ei, len(valid_env['y'])))
        for ei, test_env in enumerate(test_envs):
            print("Size of test env {}: {}".format(ei, len(test_env['y'])))
        
        assert len(train_envs) == 1 and len(valid_envs) <= 1
        train_data = (train_envs[0]['X'], train_envs[0]['y'])
        train_size = len(train_data[1])
        if len(valid_envs) > 0:
            valid_data = (valid_envs[0]['X'], valid_envs[0]['y'])
            valid_size = len(valid_data[1])
        else:
            valid_data = None
        
        ei_trainer = EITrainer(num_classes=num_classes, 
                               model=ei_model,
                               pre_optimizer=ei_pre_optimizer, 
                               optimizer=ei_optimizer, 
                               temperature=args.ei__temperature, 
                               device=args.device)
        il_trainer = ILTrainer(num_classes=num_classes,
                               model=il_model, 
                               optimizer=il_optimizer,
                               device=args.device)
    
        # ERM pretraining on EI model
        ei_trainer.pre_train(train_data=train_data,
                             valid_data=valid_data,
                             batch_size=args.ei__pre_batch_size, 
                             l2_w=ei__pre_l2_w,
                             grad_clip_norm=args.ei__pre_grad_clip_norm, 
                             iters=args.ei__pre_iters, 
                             log_times=args.ei__pre_log_times)
        
        # Joint optimization of EI and IL models
        get_inv_losses = lambda data: il_trainer.test(envs=[dict(zip(['X', 'y'], data))], 
                                                      batch_size=args.il__batch_size,
                                                      metric="losses")[0].cpu()
        get_gamma = lambda it: args.ei__gamma * ((it - 1) / (args.joint_iters - 1))
        get_penalty_w = lambda it: args.il__penalty_w if it > args.il__anneal_joint_iters else args.il__anneal_penalty_w
        for it in range(1, args.joint_iters + 1):
            print("\n------------------------- Iter {} -------------------------".format(it))
            gamma, penalty_w = get_gamma(it), get_penalty_w(it)
            print("* EI: gamma = {}\n* IL: penalty_w = {}".format(gamma, penalty_w))
            assert len(train_data[1]) == train_size 
            assert valid_data is None or len(valid_data[1]) == valid_size
        
            # Training of EI model
            ei_trainer.train(train_data=train_data,
                             valid_data=valid_data,
                             inv_losses=get_inv_losses(train_data),
                             batch_size=args.ei__batch_size, 
                             iters=args.ei__iters, 
                             beta=args.ei__beta,
                             gamma=gamma,
                             l2_w=ei__l2_w,
                             envw_thres=args.ei__envw_thres,
                             grad_clip_norm=args.ei__grad_clip_norm,
                             log_times=args.ei__log_times)
            
            ei_trainer.analyze(data=train_data, 
                               batch_size=args.ei__batch_size)
            
            # Environment inference by EI model
            train_envs, train_env_confs = ei_trainer.infer_and_pack(infer_data=train_data, 
                                                                    batch_size=args.ei__batch_size)
            if valid_data is not None:
                valid_envs, _ = ei_trainer.infer_and_pack(infer_data=valid_data, 
                                                          batch_size=args.ei__batch_size)
            
            # Training of IL model
            il_trainer.train(train_envs=train_envs, 
                             valid_envs=valid_envs,
                             env_confs=train_env_confs,
                             batch_size=args.il__batch_size,
                             penalty_w=penalty_w, 
                             l2_w=il__l2_w, 
                             grad_clip_norm=args.il__grad_clip_norm,
                             iters=args.il__iters,
                             log_times=args.il__log_times,
                             test_metric=test_metric)
        
            # Evaluation of IL model
            test_scores, test_avg = il_trainer.test(envs=test_envs, 
                                                    batch_size=args.il__batch_size, 
                                                    metric=test_metric, 
                                                    return_avg=True)
            test_scores = [s.cpu().numpy().round(decimals=4) for s in test_scores]
            test_avg = test_avg.cpu().item()
            print("\nTest Scores for iter {}: {} (Avg: {:.4f})".format(it, test_scores, test_avg))
            
        all_test_scores.append(test_scores)
        all_test_avgs.append(test_avg)
        print("\nFinal Test Scores for run {}: {} (Avg: {:.4f})".format(ri, test_scores, test_avg))
        
    print('\n', end='')
    all_test_scores = np.stack(all_test_scores, axis=0)
    print("Test Scores Mean: {} (Avg: {:.4f})".format(np.around(np.mean(all_test_scores, axis=0), decimals=4),
                                                      np.mean(all_test_avgs)))
    print("Test Scores Std: {} (Avg: {:.4f})".format(np.around(np.std(all_test_scores, axis=0), decimals=4),
                                                     np.std(all_test_avgs)))
