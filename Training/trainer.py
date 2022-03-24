import os
import time 
import random
import numpy as np
import warnings
from datetime import datetime

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchaudio
from torchvision import transforms
import torchaudio.transforms as atransforms


from utils import *
from logger import *
from model import *
from dataset import AudioMNISTDataset


class trainer(object):
    def __init__(self, config):
        self.config = config
        # setting the random seed if given
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
            torch.cuda.manual_seed_all(config.random_seed)
        experiment_name = "{}_I{}W{}R{}A{}_{}".format(config.model, config.ib, config.wb, config.rb, config.ab, datetime.now().strftime("%Y%m%d"))
        self.output_dir_path = os.path.join(config.experiments, experiment_name)
        # resuming the experiment from the given path
        if config.resume:
            self.output_dir_path, _ = os.path.split(config.resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)
        # some important directory paths
        self.checkpoints_dir_path = os.path.join(self.output_dir_path, "checkpoints")
        self.export_path = "./export/{}_W{}R{}/".format(config.model, config.wb, config.rb)
        if not config.dry_run:
            os.makedirs(self.checkpoints_dir_path, exist_ok=True)
            self.writer = SummaryWriter(self.output_dir_path + "/tensorboard/")

        # adding logger
        self.logger = Logger(self.output_dir_path, config.dry_run)
        self.starting_epoch = 1
        self.best_acc = 0
        
        # ---------------------------------- Datasets ------------------------------------------------
        # datasets
        if config.dataset == "AudioMNIST":
            n_fft = 400
            self.input_size = 13#n_fft//2 + 1
            self.num_classes = 10
            # Specify transformations to be applied to the raw audio
            self.transform_list = transforms.Compose([
                atransforms.Resample(48e3, 8e3),
                lambda audio: atransforms.MFCC(sample_rate=8e3, n_mfcc=self.input_size+1)(audio)[1:, :],
                lambda mfcc: (mfcc - mfcc.mean(axis=0)) / mfcc.std(axis=0),
                lambda mfcc: mfcc.transpose(1, 0),
            ])

            # Initialize a generator for a local version of FSDD
            dataset = AudioMNISTDataset(path=config.datadir, 
                                        transforms_train=self.transform_list,
                                        transforms_test=self.transform_list)

            # Create two Torch datasets for a train-test split from the generator
            train_set, test_set = dataset.train_test_split(test_size=0.2)
        else:
            raise Exception("Dataset not supported: {}".format(config.dataset))
        # dataloaders
        self.train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, 
                                       drop_last=False, pin_memory=False, num_workers=config.num_workers,
                                       collate_fn=collate_fn)

        self.test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, 
                                      drop_last=False, pin_memory=False, num_workers=config.num_workers,
                                      collate_fn=collate_fn)
        
        # ---------------------------------- Device ------------------------------------------------
        # setting up the GPU if we are running on a GPU
        if config.gpus is not None:
            config.gpus = [int(i) for i in config.gpus.split(',')]
            self.device = 'cuda:' + str(config.gpus[0])
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)
        

        # ---------------------------------- Model ------------------------------------------------
        if config.model == "QORNN":
            self.model = QORNN_Model(self.input_size, None, 128, self.num_classes, config).to(self.device)
            p_orth = self.model.rnn.recurrent_kernel
            orth_params = p_orth.parameters()
            non_orth_params = (
                param for param in self.model.parameters() if param not in set(p_orth.parameters())
            )
        else:
            raise Exception("Model not supported: {}".format(config.model))        
        # ---------------------------------- Loss/optimizers ------------------------------------------------
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam([{"params": non_orth_params}, 
                                     {"params": orth_params, "lr": config.lr_orth}], 
                                    lr=config.lr, betas=(0.9, 0.999))
        milestones = [int(i) for i in config.milestones.split(',')]
        if config.scheduler is None:
            self.scheduler = None
        elif config.scheduler == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
            self.logger.log.info("Using {} scheduler.".format(config.scheduler))
        else:
            raise Exception("Scheduler not supported: {}".format(config.scheduler))   

        # ---------------------------------- Resume ------------------------------------------------
        # Resuming the model if given
        if config.resume:
            self.logger.log.info("Loading model checkpoint at: {}".format(config.resume))
            package = torch.load(config.resume, map_location=self.device)
            model_state_dict = package["model_state_dict"]
            self.model.load_state_dict(model_state_dict, strict=True)
        # resuming the optimizer if we are resuming the training session
        if config.resume and not config.evaluate and not config.export:
            if "opt_state_dict" in package.keys():
                self.optimizer.load_state_dict(package["opt_state_dict"])
            if "epoch" in package.keys():
                self.starting_epoch = package["epoch"]
            if "best_acc" in package.keys():
                self.best_acc = package["best_acc"]
            if "config" in package.keys():
                temp_config = package["config"]
                if temp_config.wb != config.wb or temp_config.rb != config.rb or temp_config.ib != config.ib:
                    self.best_acc = 0

        # resuming the schedular if any
        if config.resume and not config.evaluate and self.scheduler is not None:
            self.scheduler.last_epoch = package["epoch"] - 1

        # ---------------------------------- Training Variables ------------------------------------------------
        
        # printing all the hyperparamters for the record
        for key in config.keys():
            self.logger.log.info("{} = {}".format(key, config[key]))


    # function to save the model at checkpoint
    def checkpoint(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.log.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            "model_state_dict" : self.model.state_dict(),
            "opt_state_dict" : self.optimizer.state_dict(),
            "epoch" : epoch + 1,
            "config" : self.config,
            "best_acc" : self.best_acc,
            }, best_path)

    def train_model(self):
        # iterate through all epochs
        for epoch in range(self.starting_epoch, self.config.epochs+1):
            self.logger.log.info("=="*25 + "Training" + "=="*25)
            # turning the train mode
            self.model.train()
            self.criterion.train()
            # meters to track losses
            epoch_meters = TrainingEpochMeters()
            # iterate over all batches of data
            for i, data in enumerate(self.train_loader):
                (input, _, lengths, target) = data
                # moving input to GPU
                input = input.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)                                   
                # training the models
                pred = self.model(input)
                loss = self.criterion(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                # ======================= Quantization =========================
                for p in list(self.model.parameters()):
                    if hasattr(p,'org'):
                        p.data.copy_(p.org)
                self.optimizer.step()
                for p in list(self.model.parameters()):
                    if hasattr(p,'org'):
                        # p.org.copy_(p.data)
                        p.org.copy_(p.data.clamp_(-1, 1))
                # ======================= Quantization =========================

                epoch_meters.loss.update(loss.item())
                epoch_meters.accuracy.update(self.model.correct(pred, target), n=self.config.batch_size)
                # logging
                if i % int(self.config.log_freq) == 0 or i == len(self.train_loader) -1:
                    self.logger.training_batch_cli_log(epoch_meters, epoch, self.config.epochs, i+1, len(self.train_loader))
            # evaluating the generator
            test_loss, test_acc = self.eval_model()
            if self.scheduler is not None:
                self.scheduler.step()
            
            name = "checkpoint_I{}W{}R{}A{}.tar".format(self.config.ib, self.config.wb, self.config.rb, self.config.ab)
            if test_acc >= self.best_acc:
                self.best_acc = test_acc
                name = "best_I{}W{}R{}A{}.tar".format(self.config.ib, self.config.wb, self.config.rb, self.config.ab)
            
            # save the model
            if not self.config.dry_run:
                self.checkpoint(epoch, name)
                self.writer.add_scalar('aa.Loss/train', epoch_meters.loss.avg, epoch)
                self.writer.add_scalar('aa.Loss/test', test_loss, epoch)
                self.writer.add_scalar('bb.Accuracy/train', epoch_meters.accuracy.avg, epoch)
                self.writer.add_scalar('bb.Accuracy/test', test_acc, epoch)

    # function to evaluate model
    def eval_model(self):
        #  switching to the eval mode
        self.logger.log.info("=="*25 + "Evaluation" + "=="*25)
        self.model.eval()
        self.criterion.eval()
        eval_meters = EvalEpochMeters() 
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                (input, _, lengths, target) = data
                input = input.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                pred = self.model(input)
                #compute loss
                loss = self.criterion(pred, target)
                eval_meters.loss.update(loss.item())
                eval_meters.accuracy.update(self.model.correct(pred, target), n=self.config.batch_size)
                #Eval batch ends
                if i % int(self.config.log_freq//2) == 0 or i == len(self.test_loader) -1:
                    self.logger.eval_batch_cli_log(eval_meters, i+1, len(self.test_loader))
        return eval_meters.loss.avg, eval_meters.accuracy.avg


    def export(self):
        self.model.eval()
        
        os.makedirs(self.export_path, exist_ok=True)
        mfcc_no = 0
        # export test input
        mfccs, files, lengths, labels = next(iter(self.test_loader))
        mfcc, file_name, length, label = Quantize(mfccs[mfcc_no], self.config.ib), files[mfcc_no], lengths[mfcc_no], labels[mfcc_no]

        with torch.no_grad():
            pred = self.model(mfcc[None, ...])
            pred = torch.argmax(pred).item()
        print(f"Label = {label}, Pred = {pred}")
        
        mfcc = np.concatenate((mfcc.numpy(), np.zeros((mfcc.shape[0],3))), axis=-1)
        np.savetxt(self.export_path + f"/test_image.txt", mfcc.reshape(-1), fmt="%.8f")
        self.logger.log.info("Test Image exported at {}".format(self.export_path + f"/test_image.txt"))

        # export model
        self.model.export(path=self.export_path)
        self.logger.log.info("Exporting Model at {}".format(self.export_path))


















