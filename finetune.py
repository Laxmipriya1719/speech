#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13, 2023
@author: lab-chen.weidong
"""

import os
import re
import json
import math
import shutil
import torch
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.metrics import accuracy_score

import utils
import models
from configs import create_workshop, get_config, dict_2_list

# Optional: confirm npy file loads properly
with open("./data/ravdess_npy/03-01-04-02-02-02-04.npy", "rb") as f:
    print("✅ Test file opened successfully.")


def remove_file_with_retry(file_path):
    # Check if the file exists before trying to remove it
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")
    else:
        print(f"File {file_path} does not exist, skipping deletion.")




class Engine():
    def __init__(self, cfg, local_rank: int, world_size: int):
        print("🚀 Engine initialized")
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = self.cfg.train.device
        self.EPOCH = self.cfg.train.EPOCH
        self.current_epoch = 0
        self.iteration = 0
        self.best_score = 0

        self.dataloader_feactory = utils.dataset.DataloaderFactory(self.cfg.dataset)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.calculate_score = utils.metric.calculate_score_classification
        self.early_stopping = utils.earlystopping.EarlyStopping(
            patience=self.cfg.train.patience,
            verbose=(self.local_rank == 0),
            higher_is_better=True
        )

        data_type = torch.int64
        self.loss_meter = utils.avgmeter.AverageMeter(device=self.device)
        self.acc_meter = utils.avgmeter.AverageMeter(device=self.device)
        self.predict_recoder = utils.recoder.TensorRecorder(device=self.device, dtype=data_type)
        self.label_recoder = utils.recoder.TensorRecorder(device=self.device, dtype=data_type)

    def config_2_json(self, jsonfile=None):
        self.jsonfile = os.path.join(self.cfg.workshop, 'config.json') if jsonfile is None else jsonfile
        with open(self.jsonfile, 'w') as f:
            json.dump(dict(self.cfg), f, indent=2)
        if self.local_rank == 0:
            print(f"💾 Configuration saved to {self.jsonfile}")

    def json_2_config(self, jsonfile=None):
        if jsonfile is not None:
            self.jsonfile = jsonfile
        assert hasattr(self, 'jsonfile'), 'Please provide the .json file path first.'
        with open(self.jsonfile, 'r') as f:
            data = json.load(f)
            self.cfg.merge_from_list(dict_2_list(data))
        if self.local_rank == 0:
            print(f"📖 Configuration loaded from {self.jsonfile}")

    def prepare_staff(self):
        print("🛠️ Preparing training components...")
        self.dataloader_train = self.dataloader_feactory.build(state='train', bs=self.cfg.train.batch_size, fold=self.fold)
        self.dataloader_test = self.dataloader_feactory.build(state='dev', bs=self.cfg.train.batch_size, fold=self.fold)

        self.cfg.model.freeze_cnn = self.cfg.train.freeze_cnn
        self.cfg.model.freeze_upstream = self.cfg.train.freeze_upstream
        model = models.vesper.VesperFinetuneWrapper(self.cfg.model).to(self.device)

        if self.cfg.train.freeze_cnn:
            for param in model.vesper.feature_extractor.parameters():
                param.requires_grad = False
        if self.cfg.train.freeze_upstream:
            for param in model.vesper.parameters():
                param.requires_grad = False

        if self.device == 'cpu':
            self.model = model
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])

        # Optimizer
        if self.cfg.train.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay
            )
        elif self.cfg.train.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                lr=self.cfg.train.lr,
                momentum=0.9,
                weight_decay=self.cfg.train.weight_decay
            )
        else:
            raise ValueError(f'Unknown optimizer: {self.cfg.train.optimizer}')

        # Scheduler with cosine annealing
        warmup_epoch = 0
        lr_max = self.cfg.train.lr
        lr_min = self.cfg.train.lr * 0.01
        T_max = self.EPOCH
        lr_lambda = lambda epoch: (epoch + 1) / warmup_epoch if epoch < warmup_epoch else \
            (lr_min + 0.5*(lr_max - lr_min)*(1.0 + math.cos((epoch - warmup_epoch)/(T_max - warmup_epoch) * math.pi))) / self.cfg.train.lr
        self.scheduler = lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lr_lambda)

        # Load model
        if self.cfg.train.load_model:
            ckpt = torch.load(self.cfg.train.load_model, map_location=self.device)
            self.model.module.load_state_dict(ckpt['model'])
            del ckpt

        # Resume training
        if self.cfg.train.resume:
            ckpt = torch.load(self.cfg.train.resume, map_location=self.device)
            self.model.module.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.scheduler.step()
            self.current_epoch = ckpt['epoch'] + 1
            self.iteration = ckpt['iteration']
            self.best_score = ckpt['best_score']
            del ckpt

        # Logging
        if self.local_rank == 0:
            self.writer = SummaryWriter(self.cfg.workshop)
            self.logger_train = utils.logger.create_logger(self.cfg.workshop, name='train')
            self.logger_test = utils.logger.create_logger(self.cfg.workshop, name='test')
            print('🧠 Main PID:', os.getpid())
        else:
            self.writer = None
            self.logger_train = None
            self.logger_test = None

        self.config_2_json()

    def train_epoch(self):
        print(f"📚 Epoch {self.current_epoch + 1}/{self.EPOCH}")
        self.model.train()
        tqdm_bar = tqdm(self.dataloader_train)
        for batch in tqdm_bar:
            inputs = batch['waveform'].to(self.device)
            labels = batch['emotion'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, padding_mask=batch.get('padding_mask', None))
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.loss_meter.update(loss.item())
            self.acc_meter.update((outputs.argmax(dim=1) == labels).float().mean().item())
            tqdm_bar.set_description(f"Loss: {self.loss_meter.avg:.4f}, Acc: {self.acc_meter.avg:.4f}")

    def run(self, fold=1):
        print("🏁 Starting training...")
        self.fold = fold
        self.prepare_staff()

        while self.current_epoch < self.EPOCH:
            self.train_epoch()
            self.scheduler.step()
            self.current_epoch += 1

            if self.early_stopping.early_stop:
                print(f"⏹️ Early stopping at epoch {self.current_epoch} (patience: {self.early_stopping.patience})")
                break

        # Save final model
        if self.local_rank == 0:
            save_path = os.path.join(self.cfg.workshop, 'best_model.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}")

        if self.cfg.dataset.have_test_set:
            self.evaluate()  # Optional if implemented


def main_worker(local_rank, cfg, world_size, dist_url):
    print(f"🎮 Worker launched: local_rank={local_rank}, world_size={world_size}")
    utils.environment.set_seed(cfg.train.seed + local_rank)

    if world_size > 1:
        backend = 'gloo' if cfg.train.device == 'cpu' else 'nccl'
        dist.init_process_group(backend=backend, init_method=dist_url, world_size=world_size, rank=local_rank)

    if cfg.model.init_with_ckpt:
        mark = re.search(r'(?<=_mark_)\w+', cfg.model.path_to_vesper)
        if mark:
            cfg.mark = mark.group() if cfg.mark is None else mark.group() + '_' + cfg.mark

    engine = Engine(cfg, local_rank, world_size)
    for fold in cfg.dataset.folds:
        create_workshop(cfg, local_rank, fold)
        engine.run(fold)

    if local_rank == 0:
        criteria = ['accuracy', 'precision', 'recall', 'F1']
        outfile = f'result/result_{cfg.model.type}_Finetune.csv'
        return_epoch = -1 if cfg.dataset.have_test_set else None
        utils.collect_result.path_to_csv(
            os.path.dirname(cfg.workshop),
            criteria,
            cfg.dataset.evaluate,
            csvfile=outfile,
            logname='test.log',
            wantlow=False,
            epoch=return_epoch
        )


def main(cfg):
    print("🚀 Main function started")
    utils.environment.visible_gpus(cfg.train.device_id)

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'

    if cfg.train.device.lower() == 'cpu':
        world_size = 1
    else:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("⚠️ No GPUs found. Switching to CPU.")
            cfg.train.device = 'cpu'
            world_size = 1

    print(f'🖥️ Device: {cfg.train.device}, World size: {world_size}, dist_url: {dist_url}')

    if world_size == 1 and cfg.train.device == 'cpu':
        main_worker(0, cfg, world_size, dist_url)
    else:
        mp.spawn(main_worker, args=(cfg, world_size, dist_url), nprocs=world_size)


if __name__ == '__main__':
    print("🔔 Launching finetune.py...")
    cfg = get_config(mode='_finetune')
    main(cfg)
