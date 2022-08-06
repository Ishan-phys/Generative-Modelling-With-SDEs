import numpy as np
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch 
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from models.ema import ExponentialMovingAverage
from sampling import get_sampler

class Step_by_Step(object):
    def __init__(self, sde, model, loss_fn, optimizer, config):
        """Class to train and save the model

        Args:
            sde: An `sde_lib.SDE` object that represents the forward SDE.
            models: A tuple of score models.
            loss_fn: the defined loss function
            optimizers: A tuple of optimizers to minimize the loss function
            config: configuration file
        """
        self.sde = sde
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        
        # Set the device and send the models to the device
        self.device = config["device"]
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Set the data loaders and writer
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.total_epochs = 0
        
        # Set the states of the two models
        self.state = self.get_state(self.model, self.optimizer)
        
        # Internal variables
        self.losses = []
        self.val_losses = []
        
        # Set the optimizer function and the training/evaluation step function
        self.optimize_fn = self._optimization_manager()
        self.train_step_fn = self._make_train_step_fn(optimizer_fn=self.optimize_fn)
        self.val_step_fn = self._make_val_step_fn()
        
    def set_loaders(self, train_loader, val_loader=None):
        """Set the data loaders for training/evaluation.

        Args:
            train_loader: the train dataset loader
            val_loader (optional): the validation dataset loader. Defaults to None.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def set_tensorboard(self, name, folder='tensorboard'):
        """This method allows the user to define a SummaryWriter to interface with TensorBoard

        Args:
            name (str): name of the file inside the folder
            folder (str, optional): the folder where file 'name' is located. Defaults to 'tensorboard'.
        """
        if not os.path.exists(f"./{folder}"):
            os.mkdir(f"./{folder}")

        suffix = datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')
        
    def get_state(self, model, optimizer):
        ema = ExponentialMovingAverage(model.parameters(), decay=self.config["ema"]["ema_rate"])
        state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
        return state
    
    def _optimization_manager(self):
        """Returns an optimize_fn based on `config`."""

        def optimize_fn(optimizer, params, step, lr=self.config["optim"]["lr"],
                        warmup=self.config["optim"]["warmup"],
                        grad_clip=self.config["optim"]["grad_clip"]):
            
            """Optimizes with warmup and gradient clipping (disabled if negative)."""
            
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)

            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

        return optimize_fn
    
    def _make_train_step_fn(self, optimizer_fn):
        """Builds function that performs a step in the training loop

        Args:
            optimizer_fn:
        """
        
        def train_step_fn(batch):
            """Running one step of training.

            Args:
                states: a dictionary containing the state of each of the two models
                batch: A mini-batch of training data.

            Returns:
                The average loss value of the mini-batch.
            """
            model = self.state["model"]
            optimizer = self.state["optimizer"]
            model.train()
            optimizer.zero_grad()
            loss = self.loss_fn(model, batch)
            loss.backward()
            self.optimize_fn(optimizer, model.parameters(), 
                                step=self.state["step"])
            self.state["step"] += 1
            self.state["ema"].update(model.parameters())
               
            return loss

        return train_step_fn
    
    def _make_val_step_fn(self):
        """Builds function that performs a step in the validation loop"""
        
        
        def perform_val_step_fn(batch):
            """Running one step of validation.

            Args:
                states: a dictionary containing the states of each of the two models
                batch: A mini-batch of evaluation data.

            Returns:
                The average loss value of the mini-batch.
            """
            model = self.state["model"]
            ema = self.state["ema"]
            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            loss = self.loss_fn(model, batch)
            ema.restore(model.parameters())
            
            return loss

        return perform_val_step_fn

    def _mini_batch_loss(self, validation=False):
        """Calculate the loss value for the mini-batch in either training or evaluation mode

        Args:
            validation (bool, optional): Set to true while training. Defaults to False.

        Returns:
            the calculated loss value
        """
        
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
            epoch_type = "Val Epoch" 
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
            epoch_type = "Train Epoch" 
            
        if data_loader is None:
            return None
            
        mini_batch_losses = []
        
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"{epoch_type}: {self.total_epochs}")
                mini_batch_loss = torch.mean(step_fn(batch))
                mini_batch_losses.append(mini_batch_loss.item())

        loss = np.mean(mini_batch_losses)
        return loss
    
    def set_seed(self, seed=-1):
        """Set the seed for reproducibility

        Args:
            seed (int, optional): Defaults to -1.
        """
        if seed >= 0:
            np.random(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False    
            torch.manual_seed(seed)
            random.seed(seed)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True   
            
        
    def train(self, n_train_iters, seed=-1):
        """Run the training loop over n_epochs

        Args:
            n_train_iters (int): the number of training steps
            seed (int, optional): Defaults to -1.
        """
        initial_step = self.total_epochs
        self.set_seed(seed)
        
        for step in range(initial_step, n_train_iters):
            
            # Keep track of the number of epochs
            self.total_epochs +=1 
            
            # Training            
            loss = self._mini_batch_loss(validation=False)        
            self.losses.append(loss)
            
            # Validation
            with torch.no_grad():
                val_loss = self._mini_batch_loss(validation=True)
                self.val_losses.append(val_loss)
                
            # Save the checkpoints 
            checkpoint_dir = self.config["training"]["ckpt_dir"]
            if step != 0 and step % self.config["training"]["check_pt_freq"] == 0 or step == n_train_iters:
                self.save_checkpoint(checkpoint_dir)
                # Print the current time and the number of epochs
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"Epochs Completed: {self.total_epochs}")
                print(f"Current Time: {current_time}")    
                
                if self.writer:
                    scalars = {'training': loss}
                    #if val_loss is not None:
                    #    scalars.update({'validation': val_loss})
                    self.writer.add_scalars(main_tag='loss', tag_scalar_dict=scalars, 
                                            global_step=step)
                    
        
        if self.writer:
            # Closes the writer
            self.writer.close()
            
    def save_checkpoint(self, ckpt_dir):
        """Builds dictionary with all elements for resuming training

        Args:
            ckpt_dir (str): directory where the checkpoint file is located
        """
        
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
            filepath = os.path.join(ckpt_dir, "checkpoint.pth")
            with open(filepath, 'w') as fp:
                pass
            
        saved_state = {
                    'model_state_dict':  self.state["model"].state_dict(),
                    'optimizer_state_dict':  self.state["optimizer"].state_dict(),
                    'ema_state_dict':    self.state["ema"].state_dict(),
                    'step':  self.state["step"],
                    'loss': self.losses,
                    'val_loss': self.val_losses,
                    'total_epochs': self.total_epochs,
                    }
        
        filepath = os.path.join(ckpt_dir, "checkpoint.pth")
        torch.save(saved_state, filepath)

    def load_checkpoint(self, filepath):
        """Loads dictionary

        Args:
            filepath (str): directory where the checkpoint file is located
        """
        loaded_states = torch.load(filepath)

        # Restore states for models and optimizers
        self.state['model'].load_state_dict(loaded_states['model_state_dict'])
        self.state['optimizer'].load_state_dict(loaded_states['optimizer_state_dict'])
        self.state['ema'].load_state_dict(loaded_states['ema_state_dict'])
        self.state['step'] = loaded_states['step']
        
        self.total_epochs = loaded_states['total_epochs']
        self.losses = loaded_states['loss']
        self.val_losses = loaded_states['val_loss']
        
    def generate_samples(self, shape, num_steps):
        """translates a given batch of image to another domain.

        Args:
            target_domain (str): specify the domain you want to translate to
            condition: the batch of images to translate
            num_steps: the number of steps for the sampler

        Returns:
            a batch of sample images
        """
        
        sampling_fn = get_sampler(sde=self.sde, shape=shape)
        
        model = self.state["model"]
        ema = self.state["ema"]
        
        # Generate the samples
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        samples = sampling_fn(model, num_steps=num_steps)
        ema.restore(model.parameters())
        
        samples = samples.detach().cpu()
        
        # Save the samples
        samples_dir = self.config["sampling"]["sample_dir"]
        if not os.path.exists(samples_dir):
            os.mkdir(samples_dir)
        
        # Save the images in the samples directory
        save_image(samples, f"{samples_dir}/samples_{target_domain}.jpg")
        
        return samples 
    
    def plot_samples(self, samples):
        """Plot the batch of samples.

        Args:
            samples: a mini_batch of samples to plot
        """
        plt.figure(figsize=(6, 6))
        grid = make_grid(samples)
        np_grid = grid.numpy().transpose((1, 2, 0))
        plt.imshow(np_grid*np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5]))
        plt.axis("off")
        
    def plot_losses(self):
        """Plot the training and the validation losses."""
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()