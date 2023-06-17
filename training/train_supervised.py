import os
import pprint

import numpy as np
import pyjson5 as json
import torch
import torch.nn as nn
import tqdm

import utilities.utils as utils
import wandb
from models import model_utils


def train_epoch(train_loader,model,optimizer,criterion,epoch,configs):
    model.train()

    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        
        if configs['mixed_precision']:
            # Creates a GradScaler once at the beginning of training.
            scaler = torch.cuda.amp.GradScaler()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs['mixed_precision']):
            image, label = batch
            image = image.to(configs['device'])
            label = label.to(configs['device'])
            
            out = model(image)
            
            loss = criterion(out,label)
            if idx%100==0:
                log_dict = {'Epoch':epoch, 'Iteration':idx,'train loss':loss.item()}
                if configs['wandb']:
                    wandb.log(log_dict)
                else:
                    print(log_dict)
            if configs['mixed_precision']:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                    loss.backward()
                    optimizer.step() 

def train(configs):
    print('='*20)
    print('Initializing classification trainer')
    print('='*20)
    metrics = utils.initialize_metrics(configs)
    checkpoint_path = utils.create_checkpoint_directory(configs)
    configs['checkpoint_path'] = checkpoint_path

    wandb.init(
                project=configs["wandb_project"],
                entity=configs["wandb_entity"],
                resume=True
            )

    criterion = nn.BCEWithLogitsLoss()
    if configs['ssl_encoder'] is None:
        base_model = model_utils.create_model(configs)
    else:
        base_model = torch.load(configs['ssl_encoder'],map_location='cpu')

    optimizer = torch.optim.AdamW(base_model.parameters(),lr=configs['lr'],weight_decay=configs['weight_decay'])
    
    #compile model (torch 2.0). Comment if using older versions of torch.
    #model = torch.compile(base_model)
    model = base_model

    train_loader, val_loader, _ = utils.prepare_supervised_learning_loaders(configs)
    model.to(configs['device'])
    best_loss = 10000.0
    for epoch in range(configs['epochs']):
        train_epoch(train_loader,model,optimizer,criterion,epoch,configs)
        val_loss = test(configs,phase='val',model=model,criterion=criterion,loader=val_loader,epoch=epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            print('New best validation loss: ',best_loss)
            print('Saving checkpoint')
            
            #Store checkpoint
            torch.save(base_model,os.path.join(configs['checkpoint_path'],'best_model.pt'))

def test(configs,phase,model=None, loader = None, criterion = None,epoch='Test'):
    
    if phase=='test':
        print('='*20)
        print('Begin Testing')
        print('='*20)
        checkpoint_path = utils.create_checkpoint_directory(configs)
        configs['checkpoint_path'] = checkpoint_path
        _, _, loader = utils.prepare_supervised_learning_loaders(configs)
        criterion = nn.BCEWithLogitsLoss()

        #Load model from checkpoint
        model = torch.load(os.path.join(configs['checkpoint_path'],'best_model.pt'),map_location=configs['device'])
        
        #compile model
        #model = torch.compile(model)

    elif phase=='val':
        print('='*20)
        print('Begin Evaluation')
        print('='*20)
    else:
         print('Uknown phase!')
         exit(3)
    
    metrics = utils.initialize_metrics(configs)
    model.to(configs['device'])
    model.eval()
    total_loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm.tqdm(loader)):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=configs['mixed_precision']):

                image, label = batch
                image = image.to(configs['device'])
                label = label.to(configs['device'])
                    
                out = model(image)

                predictions = nn.Sigmoid()(out.clone())
                predictions[predictions>=0.5] = 1.0
                predictions[predictions<0.5] = 0

                loss = criterion(out,label)
                total_loss += loss.item()
                for metric in metrics:
                    metric(out,label)
                num_samples += image.shape[0]
                
    total_loss = total_loss/num_samples
    log_dict = {'Epoch':epoch, phase + ' loss':total_loss}

    for idx,metric in enumerate(metrics):
        if metric.average!= 'none':
            log_dict[phase + ' ' + metric.average + ' ' + metric.__class__.__name__] = metric.compute()
        else:
            if phase!='val':
                scores = metric.compute()
                for idx in range(scores.shape[0]):
                    log_dict[phase + ' ' + metric.__class__.__name__ + ' Class: ' + str(idx)] = scores[idx]

    if configs['wandb']:
        wandb.log(log_dict)
    else:
        print(log_dict)
    return total_loss