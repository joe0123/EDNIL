from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from datareaders.utils import InfiniteDataLoader

class EITrainer:
    def __init__(self, num_classes, model, pre_optimizer, optimizer, temperature, device):
        if num_classes == 1:    # Regression
            self.erm_criterion = nn.MSELoss(reduction="none")
        elif num_classes == 2:  # Binary classification
            self.erm_criterion = nn.BCEWithLogitsLoss(reduction="none")
        elif num_classes > 2:   # Multi-class classification
            self.erm_criterion = nn.CrossEntropyLoss(reduction="none")
        else:
            raise ValueError
        self.num_classes = num_classes
        self.model = model
        self.model.to(device)
        self.pre_optimizer = pre_optimizer
        self.optimizer = optimizer
        self.softmin_fn = lambda losses: (-losses / temperature).softmax(dim=-1)
        self.device = device
    
    def compute_erm_losses(self, X, y, return_preds=False):
        preds = self.model(X)   # (N, E) or (N, C * E)
        assert len(preds.shape) == 2
        if self.num_classes > 2:
            preds = preds.reshape(preds.shape[0], self.num_classes, -1) # (N, C * E) -> (N, C, E)
        ys = y.reshape(-1, 1).repeat(1, preds.shape[-1])   # (N, E)
        assert ys.shape == (preds.shape[0], preds.shape[-1])
        losses = self.erm_criterion(preds, ys)  # (N, E)
        assert losses.shape == (preds.shape[0], preds.shape[-1])
        
        if return_preds:
            return losses, preds
        else:
            return losses    
    
    def compute_ed_loss(self, env_probs, envw_thres=1):
        assert envw_thres >= 1
        env_log_probs = env_probs.clamp(min=1e-10).log()
        env_mask = F.one_hot(env_log_probs.argmax(-1), env_log_probs.shape[1])
        if envw_thres > 1:
            env_counts = env_mask.sum(0)
            env_w = env_counts.max() / env_counts.clamp(min=1e-10)
            env_mask = env_mask * env_w.clamp(max=envw_thres).reshape(1, -1)
        assert env_mask.shape == env_log_probs.shape
        ed_loss = -(env_mask * env_log_probs).sum() / env_mask.sum()
        return ed_loss

    def compute_li_loss(self, y, env_probs): 
        # In terms of learning E, Ee[ D_KL[ P(Y | e) || Uniform ] ] is a non-negative term 
        # equivalent to L_LI in the paper. The difference is a constant.
        if self.num_classes < 2:
            raise NotImplementedError
        cy_e = torch.zeros(env_probs.shape[1], self.num_classes).to(self.device)
        for b in range(self.num_classes):
            cy_e[:, b] += env_probs[y.reshape(-1) == b].sum(dim=0)
        assert cy_e.sum().round() == len(y)
        py_e = cy_e / cy_e.sum(-1, keepdim=True).clamp(min=1e-10)
        assert py_e.shape == cy_e.shape
        uni = (torch.ones_like(py_e) / self.num_classes).to(self.device)
        assert (py_e.sum(-1).round() == 1).all() and (uni.sum(-1).round() == 1).all()
        li_loss = F.kl_div(uni.log(), py_e, reduction="batchmean")
        return li_loss
    
    def compute_ip_loss(self, inv_losses, env_probs):
        env_inv_losses = (env_probs * inv_losses.reshape(-1, 1)).sum(0) / env_probs.sum(0).clamp(min=1e-10)
        assert env_inv_losses.shape == (env_probs.shape[1],)
        ip_loss = (env_inv_losses - env_inv_losses.mean()).pow(2).mean()
        return ip_loss
    
    def compute_weight_norm(self):
        weight_norm = torch.tensor(0.).to(self.device)
        for w in self.model.parameters():
            weight_norm += w.norm().pow(2)
        return weight_norm
    
    def pre_train(self, train_data, valid_data, batch_size, l2_w, grad_clip_norm, iters, log_times):
        X_train, y_train = train_data
        is_full_batch = (batch_size <= 0)
        print("\nStart {}-batch ERM pretraining for {} iterations...".format("Full" if is_full_batch else "Mini", iters))
        if is_full_batch:
            train_iterator = cycle([(X_train, y_train)])
        elif torch.is_tensor(X_train):
            train_iterator = iter(InfiniteDataLoader(TensorDataset(X_train, y_train), 
                                                     batch_size=batch_size))
        else:
            raise NotImplementedError
        
        log_iters = iters // log_times
        accum_loss, accum_erm_loss = 0, 0
        for it in range(1, iters + 1):
            batch = next(train_iterator)
            if isinstance(batch, tuple) or isinstance(batch, list):
                bx, by = batch
                bx, by = bx.to(self.device), by.to(self.device)
            else:
                raise NotImplementedError
            
            self.model.train()
            erm_loss = self.compute_erm_losses(X=bx, y=by)[:, 0].mean()
            if l2_w > 0:
                weight_norm = self.compute_weight_norm()
            else:    # Not counting the number of parameters for efficiency
                weight_norm = 0
            loss = erm_loss + l2_w * weight_norm
            accum_erm_loss += erm_loss.detach().item()
            accum_loss += loss.detach().item()
            
            self.pre_optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            self.pre_optimizer.step()
            
            if it % log_iters == 0:
                accum_loss = accum_loss / log_iters
                accum_erm_loss = accum_erm_loss / log_iters
                print("Iter {} | Train Loss = {:.4f} (ERM = {:.4f})".format(it, accum_loss, accum_erm_loss), flush=True)
                if valid_data is not None:
                    spaces = ' ' * (8 + len(str(it)))
                    valid_erm_loss = self.pre_eval(data=valid_data, 
                                                   batch_size=batch_size)
                    print("{}Valid ERM Loss = {:.4f}".format(spaces, valid_erm_loss), flush=True)
                accum_loss, accum_erm_loss = 0, 0
        
        print("ERM pretraining done.")
        
        return self
    
    @torch.no_grad()
    def pre_eval(self, data, batch_size):
        X, y = data
        is_full_batch = (batch_size <= 0)
        if is_full_batch:
            dataloader = [(X, y)]
        elif torch.is_tensor(X):
            dataloader = DataLoader(TensorDataset(X, y), 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    num_workers=4)
        else:
            NotImplementedError

        total_loss = 0
        self.model.eval()
        for batch in dataloader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                bx, by = batch
                bx, by = bx.to(self.device), by.to(self.device)
            else:
                raise NotImplementedError
            loss = self.compute_erm_losses(X=bx, y=by)[:, 0].mean()
            total_loss += loss.detach().item()
        total_loss /= len(dataloader)
        return total_loss
         
    def train(self, train_data, valid_data, inv_losses, batch_size, beta, gamma, l2_w, envw_thres, grad_clip_norm, iters, log_times):
        X_train, y_train = train_data
        is_full_batch = (batch_size <= 0)
        print("\nStart {}-batch EI training for {} iterations...".format("Full" if is_full_batch else "Mini", iters))
        if is_full_batch:
            train_iterator = cycle([(X_train, y_train, inv_losses)])
        elif torch.is_tensor(X_train):
            train_iterator = iter(InfiniteDataLoader(TensorDataset(X_train, y_train, inv_losses), 
                                                     batch_size=batch_size))
        else:
            raise NotImplementedError
        
        log_iters = iters // log_times
        accum_loss, accum_ed_loss, accum_li_loss, accum_ip_loss = 0, 0, 0, 0
        for it in range(1, iters + 1):
            batch = next(train_iterator)
            if isinstance(batch, tuple) or isinstance(batch, list):
                bx, by, b_inv = batch
                bx, by, b_inv = bx.to(self.device), by.to(self.device), b_inv.to(self.device)
            else:
                raise NotImplementedError

            self.model.train()
            erm_losses = self.compute_erm_losses(X=bx, y=by)
            env_probs = self.softmin_fn(erm_losses)
            ed_loss = self.compute_ed_loss(env_probs=env_probs, envw_thres=envw_thres)
            li_loss = self.compute_li_loss(y=by, env_probs=env_probs)
            ip_loss = self.compute_ip_loss(inv_losses=b_inv, env_probs=env_probs)
            if l2_w > 0:
                weight_norm = self.compute_weight_norm()
            else:    # Not counting the number of parameters for efficiency
                weight_norm = 0
            loss = ed_loss + beta * li_loss + gamma * ip_loss + l2_w * weight_norm
            self.optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            self.optimizer.step()
            accum_loss += loss.detach().item()
            accum_ed_loss += ed_loss.detach().item()
            accum_li_loss += li_loss.detach().item()
            accum_ip_loss += ip_loss.detach().item()
            
            if it % log_iters == 0:
                accum_loss = accum_loss / log_iters
                accum_ed_loss = accum_ed_loss / log_iters
                accum_li_loss = accum_li_loss / log_iters
                accum_ip_loss = accum_ip_loss / log_iters
                train_env_sizes, train_env_avg_losses = self.eval(data=train_data, 
                                                                  batch_size=batch_size)
                if valid_data is not None:
                    valid_env_sizes, valid_env_avg_losses = self.eval(data=valid_data, 
                                                                      batch_size=batch_size) 
                print("Iter {} | Train Loss = {:.4f} (ED = {:.4f}, LI = {:.4f}, IP = {:.4f})".format(
                        it, accum_loss, accum_ed_loss, accum_li_loss, accum_ip_loss), flush=True)
                print(train_env_sizes)
                print(train_env_avg_losses)

                if valid_data is not None:
                    print(valid_env_sizes)
                    print(valid_env_avg_losses)
                accum_loss, accum_ed_loss, accum_li_loss, accum_ip_loss = 0, 0, 0, 0
        
        print("EI training done.")
        
        return self
    
    @torch.no_grad()
    def eval(self, data, batch_size):
        env_ids, losses, _ = self.infer(data=data, 
                                        batch_size=batch_size, 
                                        return_details=True)
        env_sizes = []
        env_avg_losses = []
        for ei in range(losses.shape[1]):
            env_mask = (env_ids == ei)
            env_sizes.append(env_mask.sum().item())
            env_avg_losses.append(losses[env_mask].mean(0))
        env_avg_losses = torch.stack(env_avg_losses, dim=0)
        return env_sizes, env_avg_losses
    
    @torch.no_grad()
    def infer(self, data, batch_size, return_details=False):
        X, y = data
        is_full_batch = (batch_size <= 0)
        if is_full_batch:
            dataloader = [(X, y)]
        elif torch.is_tensor(X):
            dataloader = DataLoader(TensorDataset(X, y), 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=4)
        else:
            NotImplementedError

        all_losses = []
        all_preds = []
        self.model.eval()
        for batch in dataloader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                bx, by = batch
                bx, by = bx.to(self.device), by.to(self.device)
            else:
                raise NotImplementedError
            losses, preds = self.compute_erm_losses(X=bx, y=by, return_preds=True)
            all_losses.append(losses.detach())
            all_preds.append(preds.detach())
        all_losses = torch.cat(all_losses, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        env_ids = all_losses.argmin(dim=-1)
        assert env_ids.shape == (len(y),)
        
        if return_details:
            return env_ids, all_losses, all_preds
        else:
            return env_ids
        
    @torch.no_grad()
    def infer_and_pack(self, infer_data, batch_size):
        env_ids, losses, _ = self.infer(data=infer_data, 
                                        batch_size=batch_size, 
                                        return_details=True)
        env_probs = self.softmin_fn(losses)
        X, y = infer_data
        envs = []
        env_confs = []
        for ei in range(losses.shape[1]):
            env_mask = (env_ids == ei)
            assert len(env_mask.shape) == 1
            if env_mask.sum() == 0:
                continue
            if torch.is_tensor(X):
                envs.append({'X': X[env_mask], 
                             'y': y[env_mask]})
            else:
                raise NotImplementedError
            env_confs.append(env_probs[env_mask, ei].mean())
        assert sum([len(env['y']) for env in envs]) == len(y)
        env_confs = torch.stack(env_confs)
        return envs, env_confs

    @torch.no_grad()
    def analyze(self, data, batch_size):
        env_ids, losses, preds = self.infer(data=data, 
                                            batch_size=batch_size, 
                                            return_details=True)
        X, y = data
        losses, preds = losses.cpu(), preds.cpu()
        if self.num_classes <= 2:
            corrects = ((preds >= 0).int() == y.reshape(-1, 1).repeat(1, preds.shape[-1]).int()).float()
        else:   # "Classes" always sits in dimension 1
            corrects = (preds.argmax(1).int() == y.reshape(-1, 1).repeat(1, preds.shape[-1]).int()).float()
        env_probs = self.softmin_fn(losses)
        labels = y.reshape(-1).cpu().numpy()
        
        accum_count = 0
        for ei in range(losses.shape[1]):
            mask = (env_ids == ei).cpu().numpy()
            count = mask.sum()
            if count == 0:
                continue
            accum_count += count
            label_dist = list(np.unique(labels[mask], return_counts=True))
            label_dist[1] = label_dist[1].astype(float) / count
            print('\n')
            print("Statistics for env {}:".format(ei))
            print("Count:", count)
            print("Env avg losses:", losses[mask].mean(dim=0))
            print("Accuracies:", corrects[mask].mean(dim=0))
            print("Confidence:", env_probs[mask, ei].mean().item())
            print("Label distribution w.r.t. {}: {}".format(*label_dist))
        assert accum_count == len(y)
        return
