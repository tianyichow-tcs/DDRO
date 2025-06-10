from model import *
from cosine_distance import *
import numpy as np
import higher
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm  import tqdm
import os
import pandas as pd

import higher
import torch
from torch.autograd import Variable
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
n_factors_dict = {"movielens": 10, "KuaiRec": 10, "coat": 5,
                    "yahoo": 5, "netflix": 5, "arxiv":20}

def test(model, device, test_loader, criterion,verbose =False):
    test_epoch_loss = 0

    model.eval()

    for user, item, rating in test_loader:

        rating = rating.to(dtype=torch.float, device=device)
        user = user.to(dtype=torch.long, device=device)
        item = item.to(dtype=torch.long, device=device)

        prediction = model(user, item)
        loss = criterion(prediction, rating)

        test_epoch_loss += loss.item()

    test_epoch_loss /= int(len(test_loader.dataset) / test_loader.batch_size)
    if verbose:

        print('\nTest Epoch loss: {:.4f}\n'.format(
            test_epoch_loss))

    return test_epoch_loss

def get_diversity(model,dist_mat,num_users,num_items,data, topk=10,soft_tau=0.0001, mask=True, mask_mat=None, diversity_target="topk-mat"):
    """
        model: recommendation model.
        dist_mat: distance matrix
        data: used to generate the mask for top-k diversity, if training dataset, 
    """

    diversity_func = CosineDistanceDiversity(top_k=topk,sort_tau=soft_tau)
    user_indices = torch.arange(num_users, device=device).unsqueeze(1).repeat(1, num_items).view(-1)
    item_indices = torch.arange(num_items, device=device).repeat(num_users)
    predictions = model(user_indices, item_indices)  # Shape: [num_users * num_items]
    R = predictions.view(num_users, num_items)

    if mask:
        R  = mask_mat*R
    approx_div,true_div = diversity_func(R,dist_mat)
    approx_diversity_score = torch.mean(approx_div)
    acc_diversity_score = torch.mean(true_div)
    return approx_diversity_score, acc_diversity_score

def baseline(dataset_name, dist_mat, num_users,num_items,  train_loader,val_loader,test_loader, top_k=10,iters=10, model_type="nn",verbose=False,initialization='opt'):
    if model_type=="nn":
        model = NCFModel(num_users, num_items,n_factors_dict[dataset_name]).to(device)
    else:
        model = NMFModel(num_users, num_items, k=n_factors_dict[dataset_name]).to(device)

    model_folder = f"models/model_init/{dataset_name}/"
    model_path = os.path.join(model_folder, f"{model_type}_{dataset_name}_{initialization}.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    best_test_loss = np.inf
    count_worst_loss = 0
    count_for_break = 5
    train_loss,val_loss, test_loss = [], [], []

    for epoch in range(1,iters+1):
        model.train()
        train_epoch_loss = []
        for users, items, ratings in train_loader:
            ratings = ratings.to(dtype=torch.float, device=device)
            users = users.to(dtype=torch.long, device=device)
            items = items.to(dtype=torch.long, device=device)
            preds = model(users, items)
            loss = criterion(preds, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.cpu().detach().numpy())
        train_loss.append(np.mean(train_epoch_loss))
        val_epoch_loss = test(model, device, val_loader, criterion)
        val_loss.append(val_epoch_loss)
        test_epoch_loss = test(model, device, test_loader, criterion)
        test_loss.append(test_epoch_loss)

        if test_epoch_loss < best_test_loss:
            best_test_loss = test_epoch_loss
            count_worst_loss = 0
            print("Best model saved")
            torch.save(model.state_dict(), model_path)
        
        if verbose:
            approx_diversity_score,acc_diversity_score = get_diversity(model,dist_mat,num_users,num_items,train_loader.dataset, topk=top_k,soft_tau=0.0001,mask=True)
            print(f"[{epoch}] Trainning Loss (no weights): {np.mean(train_epoch_loss):.4f}",f"Validation Loss: {val_epoch_loss:.4f}", f"Test Loss: {test_epoch_loss:.4f}",f"approx_div:{approx_diversity_score.cpu().detach().numpy():.4f}",f"acc_div:{acc_diversity_score.cpu().detach().numpy():.4f}")
        else:
            print(f"[{epoch}] Trainning Loss (no weights): {np.mean(train_epoch_loss):.4f}",f"Validation Loss: {val_epoch_loss:.4f}", f"Test Loss: {test_epoch_loss:.4f}")

def baseline_diversity(model_init,dataset_name, dist_mat, num_users,train_loader,val_loader,test_loader, num_items, top_k,iters=10,beta=0.7,learning_rate = 0.01,model_type ="mf",mask=True, diversity_target="topk-mat",train_strategy="finetune"):
    mask_mat=None
    if diversity_target =="topk-mat":
        data = train_loader.dataset
        mask_mat = torch.ones((num_users, num_items), dtype=torch.float).to(device)
        for u, i, _ in data:
            mask_mat[u, i] = 0
    elif diversity_target =="topk-val":
        data = val_loader.dataset
        mask_mat = torch.zeros((num_users, num_items), dtype=torch.float).to(device)
        for u, i, _ in data:
            mask_mat[u, i] = 1


    if model_type =="nn":
        model = NCFModel(num_users, num_items,n_factors_dict[dataset_name]).to(device)
    else:
        model = NMFModel(num_users, num_items, n_factors_dict[dataset_name]).to(device)
        
    model.load_state_dict(model_init.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    topk =top_k
    soft_tau=0.0001
    loss_fn_div = CosineDistanceDiversity(top_k=topk,sort_tau=soft_tau)
    log_dict = {
    "epoch": [],
    "F_loss":[],
    "train_loss": [],
    "val_loss": [],
    "test_loss": [],
    "approx_div": [],
    "acc_div": []
}
    for epoch in range(1,iters+1):
        losses = []
        F_losses = []
        model.train()
        for users, items, ratings in train_loader:
            ratings = ratings.to(dtype=torch.float, device=device)
            users = users.to(dtype=torch.long, device=device)
            items = items.to(dtype=torch.long, device=device)

            preds = model(users, items)
            loss = F.mse_loss(preds, ratings)

            approx_diversity_score,acc_diversity_score = get_diversity(model,dist_mat,num_users,num_items, data, topk=top_k, soft_tau=0.0001,mask=mask,mask_mat=mask_mat, diversity_target=diversity_target)

            F_loss = beta * loss - (1-beta)*approx_diversity_score
            optimizer.zero_grad()
            F_loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            F_losses.append(F_loss.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            val_users, val_items,val_ratings = next(iter(val_loader))
            val_ratings = val_ratings.to(dtype=torch.float, device=device)
            val_users = val_users.to(dtype=torch.long, device=device)
            val_items = val_items.to(dtype=torch.long, device=device)
            val_preds = model(val_users, val_items)
            val_loss = F.mse_loss(val_preds, val_ratings)
            test_users, test_items,test_ratings =next(iter(test_loader))
            test_ratings = test_ratings.to(dtype=torch.float, device=device)
            test_users = test_users.to(dtype=torch.long, device=device)
            test_items = test_items.to(dtype=torch.long, device=device)
            test_preds = model(test_users, test_items)
            loss_f_test = F.mse_loss(test_preds, test_ratings)


            approx_diversity_score,acc_diversity_score = get_diversity(model,dist_mat,num_users,num_items, data, topk=top_k,soft_tau=0.0001,mask=mask,mask_mat=mask_mat, diversity_target=diversity_target)


        log_dict["epoch"].append(epoch)
        log_dict["F_loss"].append(float(np.mean(F_losses)))
        log_dict["train_loss"].append(float(np.mean(losses)))
        log_dict["val_loss"].append(float(val_loss.item()))
        log_dict["test_loss"].append(float(loss_f_test.item()))
        log_dict["approx_div"].append(float(approx_diversity_score.cpu().detach().numpy()))
        log_dict["acc_div"].append(float(acc_diversity_score.cpu().detach().numpy()))
        if epoch % 5==0:
            model_path = "models_dict/checkpoints/{}/{}/{}_{}_{}_with_{}_beta{}_epoch{}_topk{}.pth".format(train_strategy,dataset_name,dataset_name,train_strategy,model_type,"algo-base",beta,epoch,top_k)
            torch.save(model.state_dict(), model_path)
            print("save model")
        print(f"[{epoch}] Trainning Loss (no weights): {np.mean(losses):.4f}",f"Validation Loss: {val_loss.item():.4f}", f"Test Loss: {loss_f_test.item():.4f}",f"approx_div:{approx_diversity_score.cpu().detach().numpy():.4f}",f"acc_div:{acc_diversity_score.cpu().detach().numpy():.4f}")
    
    result_path = "models_dict/logs/{}/{}_{}_{}_with_{}_beta{}_epoch{}_topk{}.pkl".format(dataset_name,dataset_name,train_strategy,model_type,"algo-base",beta,epoch,topk)  
    with open(result_path, 'wb') as f:
        pickle.dump(log_dict, f)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,dtype=torch.float, requires_grad=requires_grad)

def meta_training(model_init,dataset_name, topk,dist_mat,num_users,num_items,train_loader,val_loader,test_loader, num_iter=100,beta=0.7, learning_rate=0.01,model_type="mf",mask=True, diversity_target="topk-mat",train_strategy="finetune"):
    mask_mat=None
    if diversity_target =="topk-mat":
        data = train_loader.dataset
        mask_mat = torch.ones((num_users, num_items), dtype=torch.float).to(device)
        for u, i, _ in data:
            mask_mat[u, i] = 0
    elif diversity_target =="topk-val":
        data = val_loader.dataset
        mask_mat = torch.zeros((num_users, num_items), dtype=torch.float).to(device)
        for u, i, _ in data:
            mask_mat[u, i] = 1
            

    if model_type =="nn":
        model = NCFModel(num_users, num_items,n_factors_dict[dataset_name]).to(device)
    else:
        model = NMFModel(num_users, num_items, n_factors_dict[dataset_name]).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    div_losses = []
    approx_diversity_scores = []
    acc_diversity_scores = []
    test_losses = []

    model.load_state_dict(model_init.state_dict())
    criterion = torch.nn.MSELoss()
    inner_opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1,num_iter+1):
        model.train()
        loss_org_epoch = []
        loss_div_epoch = []
        approx_socres = []
        acc_scores  = []
        for users, items, ratings in train_loader:
            ratings = ratings.to(dtype=torch.float, device=device)
            users = users.to(dtype=torch.long, device=device)
            items = items.to(dtype=torch.long, device=device)

            # inner loop adaptation with higher
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
                # compute weighted loss
                preds = fmodel(users, items)
                # loss_fn = SELoss()
                cost = (preds - ratings) ** 2

                eps = torch.zeros(cost.size(), dtype=torch.float, device=device, requires_grad=True)
                # l_f_meta = torch.mean(cost * eps)
                l_f_meta =   (eps *cost).mean()
                diffopt.step(l_f_meta)
                
                # after one step, compute the losses
                preds_inner = fmodel(users, items)
                loss_org = criterion(preds_inner, ratings)
                loss_org_epoch.append(loss_org.item())
                # compute diversity
                approx_div_score,acc_div_score = get_diversity(fmodel,dist_mat,num_users,num_items,data, topk=topk,soft_tau=0.0001,mask=mask,mask_mat=mask_mat, diversity_target=diversity_target)
                approx_socres.append(approx_div_score.item())
                acc_scores.append(acc_div_score.item())

                loss_div = beta * loss_org - (1 - beta) * approx_div_score
                loss_div_epoch.append(loss_div.item())
                grad_eps = torch.autograd.grad(loss_div, eps, only_inputs=True, retain_graph=True)[0]

                w_tilde = torch.clamp(-grad_eps, min=0)
                norm_c = torch.sum(w_tilde)
                if norm_c != 0:
                    w = w_tilde / norm_c
                else:
                    w = w_tilde

                # Back to the original model, update using outer loss
                preds_org = model(users, items)
                cost = (preds_org-ratings)**2
                l_f = torch.sum(cost * w).mean()

                opt.zero_grad()
                l_f.backward()
                opt.step()


        val_users, val_items, val_ratings = next(iter(val_loader))
        val_users = val_users.to(dtype=torch.long, device=device)
        val_items = val_items.to(dtype=torch.long, device=device)
        val_ratings = val_ratings.to(dtype=torch.float, device=device)

        val_preds = model(val_users, val_items)
        loss_val = criterion(val_preds, val_ratings)

        test_users, test_items, test_ratings = next(iter(test_loader))
        test_users = test_users.to(dtype=torch.long, device=device)
        test_items = test_items.to(dtype=torch.long, device=device)
        test_ratings = test_ratings.to(dtype=torch.float, device=device)

        test_preds = model(test_users, test_items)
        loss_test = criterion(test_preds, test_ratings)

        print(f"[{epoch}] Loss_train:{np.mean(loss_org_epoch):.4f}", f"loss_div:{np.mean(loss_div_epoch):.4f}",
                f"val_loss:{loss_val.item():.4f}", f"test_loss:{loss_test.item():.4f}", f"approx_div_score:{np.mean(approx_socres):.4f}", f"acc_div_score:{np.mean(acc_scores):.4f}",)
        if epoch % 5==0:
            model_path = "models_dict/checkpoints/{}/{}/{}_{}_{}_with_{}_beta{}_epoch{}_topk{}.pth".format(train_strategy,dataset_name,dataset_name,train_strategy,model_type,"algo-rw",beta,epoch,topk)
            torch.save(model.state_dict(), model_path)
            print("save model")
        train_losses.append(np.mean(loss_org_epoch))
        div_losses.append(np.mean(loss_div_epoch))
        approx_diversity_scores.append(np.mean(approx_socres))
        acc_diversity_scores.append(np.mean(acc_scores))
        test_losses.append(loss_test.item())
        val_losses.append(loss_val.item())

    result_dict = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "F_loss": div_losses,
        "approx_div": approx_diversity_scores,
        "acc_div": acc_diversity_scores,
        "test_loss": test_losses
    }
    result_path = "models_dict/logs/{}/{}_{}_{}_with_{}_beta{}_epoch{}_topk{}.pkl".format(dataset_name,dataset_name,train_strategy,model_type,"algo-rw",beta,epoch,topk)  
    with open(result_path, 'wb') as f:
        pickle.dump(result_dict, f)

