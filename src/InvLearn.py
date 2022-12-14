from itertools import cycle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad

from datareaders.utils import InfiniteDataLoader

class ILTrainer:
    def __init__(self, num_classes, model, optimizer, device):
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
        self.optimizer = optimizer
        self.device = device
    
    def compute_il_loss(self, envs, env_confs, penalty_w, l2_w):
        def _irm_penalty(logits, y):
            scale = torch.tensor(1.).to(self.device).requires_grad_()
            erm_loss = self.erm_criterion(logits * scale, y).mean()
            grad_single = grad(erm_loss, [scale], create_graph=True)[0]
            penalty = grad_single.pow(2).sum()
            return penalty
        def _model_weight_norm():
            weight_norm = torch.tensor(0.).to(self.device)
            for w in self.model.parameters():
                weight_norm += w.norm().pow(2)
            return weight_norm
        
        erm_loss_avg = 0.
        penalty_avg = 0.
        assert len(envs) == len(env_confs)
        for data, env_conf in zip(envs, env_confs):
            if isinstance(data, tuple) or isinstance(data, list):
                X, y = data
                X, y = X.to(self.device), y.to(self.device)
            else:
                raise NotImplementedError
            outputs = self.model(X)
            erm_loss = self.erm_criterion(outputs, y).mean()
            penalty = _irm_penalty(outputs, y)
            erm_loss_avg += env_conf * erm_loss
            penalty_avg += env_conf * penalty
        erm_loss_avg /= env_confs.sum()
        penalty_avg /= env_confs.sum()
        
        if l2_w > 0:    # Not counting the number of parameters for efficiency
            weight_norm = _model_weight_norm()
        else:
            weight_norm = 0
        total_loss = erm_loss_avg + penalty_w * penalty_avg + l2_w * weight_norm
        total_loss /= max(penalty_w, 1)
        
        return total_loss, erm_loss_avg, penalty_avg
    
    def train(self, train_envs, valid_envs, env_confs, batch_size, 
              penalty_w, l2_w, grad_clip_norm,
              iters, log_times, test_metric):
        is_irm = (penalty_w > 0)
        is_full_batch = (batch_size <= 0)
        print("\nStart {}-batch {} for {} iterations...".format("Full" if is_full_batch else "Mini",
                                                                "IRM" if is_irm else "ERM",
                                                                iters))
        if is_full_batch:
            train_dataloaders = [cycle([(d['X'], d['y'])]) for d in train_envs]
        elif torch.is_tensor(train_envs[0]['X']):
            batch_size_per_env = batch_size // len(train_envs)
            train_dataloaders = [InfiniteDataLoader(TensorDataset(d['X'], d['y']), 
                                                    batch_size=batch_size_per_env) for d in train_envs]
        else:
            raise NotImplementedError
        train_iterator = zip(*train_dataloaders)
        
        log_iters = iters // log_times
        accum_loss, accum_erm_loss, accum_penalty = 0, 0, 0
        for it in range(1, iters + 1):
            batch_envs = next(train_iterator)
            
            self.model.train()
            loss, erm_loss, penalty = self.compute_il_loss(envs=batch_envs, 
                                                           env_confs=env_confs,
                                                           penalty_w=penalty_w, 
                                                           l2_w=l2_w)
            accum_loss += loss.detach().item()
            accum_erm_loss += erm_loss.detach().item()
            accum_penalty += penalty.detach().item()
            
            self.optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            self.optimizer.step()
            
            if it % log_iters == 0:
                accum_loss = accum_loss / log_iters
                accum_erm_loss = accum_erm_loss / log_iters
                accum_penalty = accum_penalty / log_iters
                train_summary = "Passed for efficiency"
                valid_scores, valid_avg = self.test(envs=valid_envs, 
                                                    batch_size=batch_size, 
                                                    metric=test_metric,
                                                    return_avg=True)
                valid_scores = [s.cpu().numpy().round(decimals=4) for s in valid_scores]
                valid_avg = valid_avg.cpu().item()
                valid_summary = "{} (Avg: {:.4f})".format(valid_scores, valid_avg)
                spaces = ' ' * (8 + len(str(it)))
                print("Iter {} | Train Loss = {:.4f} (ERM = {:.4f}, Penalty = {:.4f})\n" \
                      "{}Train Scores = {}\n{}Valid Scores = {}".format(
                        it, accum_loss, accum_erm_loss, accum_penalty, \
                        spaces, train_summary, \
                        spaces, valid_summary), flush=True)
                accum_loss, accum_erm_loss, accum_penalty = 0, 0, 0
        
        if is_irm:
            print("Invariant Learning done.")
        else:
            print("ERM Training done.")
        
        return self
    
    @torch.no_grad()
    def test(self, envs, batch_size, metric, return_avg=False):
        assert not (return_avg == True and metric == "losses")
        
        is_full_batch = (batch_size <= 0)
        results = []
        avg, total_num = torch.tensor(0.).to(self.device), 0
        for d in envs:
            if is_full_batch: 
                dataloader = [(d['X'], d['y'])]
            elif torch.is_tensor(d['X']):
                dataloader = DataLoader(TensorDataset(d['X'], d['y']), 
                                        batch_size=batch_size, 
                                        shuffle=False,
                                        num_workers=4)
            else:
                NotImplementedError
            outputs = []
            labels = []
            self.model.eval() 
            for data in dataloader:
                if isinstance(data, tuple) or isinstance(data, list):
                    X, y = data
                    X, y = X.to(self.device), y.to(self.device)
                else:
                    raise NotImplementedError
                outputs.append(self.model(X))
                labels.append(y)
            outputs = torch.cat(outputs, dim=0)
            labels = torch.cat(labels, dim=0)
            if metric == "loss":
                result = self.erm_criterion(outputs, labels).mean().detach()
            elif metric == "losses":
                result = self.erm_criterion(outputs, labels).reshape(-1).detach()
                assert result.shape == (labels.shape[0],)
            elif metric == "acc":
                if self.num_classes <= 2:
                    result = ((outputs >= 0).reshape(-1).int() == labels.reshape(-1).int()).float().mean().detach()
                else:   # "Classes" always sits in dimension 1
                    result = (outputs.argmax(1).reshape(-1).int() == labels.reshape(-1).int()).float().mean().detach()
            elif metric == "mse":
                result = (outputs.reshape(-1) - labels.reshape(-1)).pow(2).mean().detach()
            else:
                raise NotImplementedError
            results.append(result)
            if return_avg:
                avg += result * labels.shape[0]
                total_num += labels.shape[0]

        if return_avg:
            avg /= max(1e-10, total_num)
            return results, avg
        else:
            return results
