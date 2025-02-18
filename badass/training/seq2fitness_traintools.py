import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
from scipy.stats import spearmanr
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from badass.utils.sequence_utils import create_absolute_sequences_list, convert_rel_seqs_to_tensors, pad_rel_seq_tensors_with_nan, rel_sequences_to_dict
from badass.models.seq2fitness_models import ProteinFunctionPredictor_with_probmatrix

class ModelTrainer():
    """
    A class for training and evaluating a machine learning model.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. Default is AdamW.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Default is None.
        model_params (dict, optional): Dictionary of model parameters. Default is None.
        verbose (bool, optional): Whether to print verbose output. Default is True.
        save_interval (int, optional): Interval for saving model checkpoints. Default is 500.

    Attributes:
        model (nn.Module): The model to be trained.
        model_checkpoints (ModelCheckpoint): Model checkpoint manager.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        model_params (dict): Dictionary of model parameters.
        criteria (dict): Dictionary of loss functions and weights for each task.
        verbose (bool): Whether to print verbose output.
        losses (dict): Dictionary to store training and validation losses.
        spearman (dict): Dictionary to store Spearman correlations for each task.
        epoch (int): The current epoch number.
        save_interval (int): Interval for saving model checkpoints.
        task_names (list): List of task names.
        num_gpus (int): Number of available GPUs.
        devices (list): List of available devices (GPUs or CPU).
        batch_converter (callable): Batch converter for the model.

    Methods:
        move_to_device(batch, device):
            Moves the batch of data to the specified device.
        prepare_batch(batch, device):
            Prepares a batch of data for the model.
        train_for_one_epoch(trainloader):
            Trains the model for one epoch.
        evaluate(valloader):
            Evaluates the model on the validation set.
        train(trainloader, valloader, max_epochs, stop_patience, save_path, filename):
            Trains the model across multiple epochs.
        plot_losses(start_epoch, stop_epoch, savepath, logscale):
            Plots training and validation losses.
        plot_spearman(start_epoch, stop_epoch, savepath):
            Plots Spearman correlations for each task over epochs.
    """
    def __init__(self, model, optimizer=None, scheduler=None, model_params=None, verbose=True, save_interval=500):
        self.model = model
        self.model_checkpoints = ModelCheckpoint(max_size=1, model_params=model_params)
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(model.parameters())
        self.scheduler = scheduler
        self.model_params = model_params
        criteria = self.model_params['task_criteria']
        self.criteria = criteria if criteria is not None else {'default': {'loss': torch.nn.MSELoss(), 'weight': 1.0}}
        self.verbose = verbose
        self.losses = {'train': [], 'val': [], 'corr': []}
        self.spearman = {task: [] for task in self.criteria.keys()}
        self.epoch = 0
        self.save_interval = save_interval
        self.task_names = list(criteria.keys())  # Extract task names directly from criteria
        self.num_gpus = torch.cuda.device_count()  # Needed to handle bypassing DataParallel for small batches
        if torch.cuda.is_available():
            self.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else:
            self.devices = [torch.device('cpu')]
        self.batch_converter = model.alphabet.get_batch_converter()
        # Wrap the model in DataParallel
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            print(f"We'll use {torch.cuda.device_count()} GPUs through DataParallel.")
        self.model.to(self.devices[0])  # Move the model to the primary device first

    def move_to_device(self, batch, device):
        """
        Moves the batch of data to the specified device.

        Args:
            batch (dict): Batch of data.
            device (torch.device): Device to move the data to.

        Returns:
            dict: Batch of data moved to the specified device.

        Example:
            batch = self.move_to_device(batch, torch.device('cuda'))
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):  # In case of nested dictionaries
                batch[key] = {k: v.to(device) for k, v in value.items() if isinstance(v, torch.Tensor)}
        return batch

    def prepare_batch(self, batch, device):
        """
        Prepares a batch of data for the model.

        Args:
            batch (dict): Batch of data.
            device (torch.device): Device to move the data to.

        Returns:
            tuple: Prepared batch tokens and relative sequence tensors padded.

        Example:
            batch_tokens, rel_seqs_tensors_padded = self.prepare_batch(batch, torch.device('cuda'))
        """
        # Grab the sequences from the batch
        rel_seqs = batch['sequence']
        # Convert relative sequences to a list of dictionaries
        rel_seqs_list_of_dicts = list(rel_sequences_to_dict(rel_seqs, sep='-').values())
        # Convert relative sequences to absolute sequences
        abs_seqs = create_absolute_sequences_list(rel_seqs_list_of_dicts, self.model_params['ref_seq'])
        # Tokenize the absolute sequences using the batch converter
        batch_labels, batch_strs, batch_tokens = self.batch_converter([(str(i), seq) for i, seq in enumerate(abs_seqs)])
        batch_tokens = batch_tokens.to(device)
        # Convert relative sequences to tensors and pad them
        rel_seqs_tensors = convert_rel_seqs_to_tensors(rel_seqs_list_of_dicts)
        rel_seqs_tensors_padded = pad_rel_seq_tensors_with_nan(rel_seqs_tensors)
        return batch_tokens, rel_seqs_tensors_padded

    def train_for_one_epoch(self, trainloader):
        """
        Trains the model for one epoch.

        Args:
            trainloader (DataLoader): DataLoader for the training set.

        Returns:
            float: Average training loss for the epoch.

        Example:
            train_loss = self.train_for_one_epoch(trainloader)
        """
        self.model.train()
        device = next(self.model.parameters()).device
        total_loss = 0.0
        sum_weights = 0.0
        all_sequences = []
        all_scores = []

        for batch in trainloader:
            self.optimizer.zero_grad()
            batch = self.move_to_device(batch, device)
            batch_tokens, rel_seqs_tensors_padded = self.prepare_batch(batch, device)
            labels = batch['labels']

            if len(labels) < self.num_gpus:
                outputs = self.model.module(batch_tokens, rel_seqs_tensors_padded)  # Bypass DataParallel
            else:
                outputs = self.model(batch_tokens, rel_seqs_tensors_padded)

            batch_loss = 0.0
            batch_weights = 0.0
            all_sequences.extend(batch['sequence'])
            all_scores.append(outputs)

            for task, task_info in self.criteria.items():
                criterion = task_info['loss']
                weight = task_info.get('weight', 1.0)
                if task in outputs:
                    task_targets = labels[task].view(-1, 1).float()
                    task_outputs = outputs[task]
                    valid_mask = ~torch.isnan(task_targets)
                    if valid_mask.any():
                        valid_targets = task_targets[valid_mask]
                        valid_outputs = task_outputs[valid_mask]
                        valid_count = valid_targets.size(0)
                        task_loss = criterion(valid_outputs, valid_targets)
                        batch_w = weight * valid_count
                        batch_loss += task_loss * batch_w
                        batch_weights += batch_w

            if batch_weights > 0:
                batch_loss = batch_loss.float() / batch_weights
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss.item() * batch_weights
                sum_weights += batch_weights

        return total_loss / sum_weights if sum_weights > 0 else 0


    def evaluate(self, valloader):
        """
        Evaluates the model on the validation set.

        Args:
            valloader (DataLoader): DataLoader for the validation set.

        Returns:
            tuple: Validation loss, Spearman correlations, and weighted Spearman correlation.

        Example:
            val_loss, spearman_corr, weighted_spearman = self.evaluate(valloader)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = 0.0
        sum_weights = 0.0
        all_sequences = []
        all_scores = []
        accumulated_predictions = {task: [] for task in self.task_names}
        accumulated_targets = {task: [] for task in self.task_names}

        with torch.no_grad():
            for batch in valloader:
                batch = self.move_to_device(batch, device)
                batch_tokens, rel_seqs_tensors_padded = self.prepare_batch(batch, device)
                labels = batch['labels']


                if len(labels) < self.num_gpus:
                    outputs = self.model.module(batch_tokens, rel_seqs_tensors_padded)  # Bypass DataParallel
                else:
                    outputs = self.model(batch_tokens, rel_seqs_tensors_padded)

                batch_loss = 0.0
                batch_weights = 0.0
                all_sequences.extend(batch['sequence'])
                all_scores.append(outputs)

                for task, task_info in self.criteria.items():
                    criterion = task_info['loss']
                    weight = task_info.get('weight', 1.0)
                    if task in outputs:
                        task_targets = labels[task].view(-1, 1).float()
                        task_outputs = outputs[task]
                        valid_mask = ~torch.isnan(task_targets)

                        if valid_mask.any():
                            valid_targets = task_targets[valid_mask]
                            valid_outputs = task_outputs[valid_mask]
                            valid_count = valid_targets.size(0)
                            task_loss = criterion(valid_outputs, valid_targets)
                            batch_w = weight * valid_count
                            batch_loss += task_loss * batch_w
                            batch_weights += batch_w
                            accumulated_predictions[task].extend(valid_outputs.cpu().numpy())
                            accumulated_targets[task].extend(valid_targets.cpu().numpy())

                total_loss += batch_loss.item()
                sum_weights += batch_weights

        spearman_correlations = {}
        for task in self.task_names:
            if accumulated_predictions[task] and accumulated_targets[task]:
                try:
                    spearman_correlations[task] = compute_spearman_correlation(accumulated_predictions[task], accumulated_targets[task])
                    print(f"Spearman correlation for {task}: {spearman_correlations[task]:.4f}")
                except Exception as e:
                    print(f"Error computing Spearman correlation for {task}: {e}")
                    spearman_correlations[task] = float('nan')

        weighted_spearman = sum((spearman_correlations[task] * self.criteria[task].get('weight', 1.0)
                                 for task in self.task_names if spearman_correlations[task] is not None))
        sum_w = sum((self.criteria[task].get('weight', 1.0)
                     for task in self.task_names if spearman_correlations[task] is not None))

        if self.model_checkpoints.save_model_predictions:
            flattened_scores = {task: [] for task in self.task_names}
            for batch_scores in all_scores:
                for task in self.task_names:
                    if task in batch_scores:
                        flattened_scores[task].extend(batch_scores[task].cpu().numpy())
            return (total_loss / sum_weights if sum_weights > 0 else 0,
                    spearman_correlations,
                    weighted_spearman / sum_w if sum_w > 0 else 0,
                    all_sequences,
                    flattened_scores)
        else:
            return (total_loss / sum_weights if sum_weights > 0 else 0,
                    spearman_correlations,
                    weighted_spearman / sum_w if sum_w > 0 else 0)


    def train(self, trainloader, valloader=None, max_epochs=100, stop_patience=1500, save_path=None, filename=""):
        """
        Trains the model across multiple epochs, with optional validation, early stopping, and model checkpointing.

        Args:
            trainloader (DataLoader): DataLoader for the training set.
            valloader (DataLoader, optional): DataLoader for the validation set. Default is None.
            max_epochs (int, optional): Maximum number of epochs to train. Default is 100.
            stop_patience (int, optional): Patience for early stopping. Default is 1500.
            save_path (str, optional): Path to save model checkpoints. Default is None.
            filename (str, optional): Prefix for the checkpoint filenames. Default is "".

        Example:
            self.train(trainloader, valloader, max_epochs=100, stop_patience=1500, save_path="checkpoints/")
        """
        print(f"Will save models to {save_path}")
        train_losses, val_losses = [], []
        best_val_loss = 1e9  # Initialize to a very large value
        no_improve_epochs = 0

        for epoch in range(self.epoch + 1, self.epoch + max_epochs + 1):
            # Train
            if self.model_checkpoints.save_model_predictions:
                train_loss, train_sequences, train_scores = self.train_for_one_epoch(trainloader)
            else:
                train_loss = self.train_for_one_epoch(trainloader)
            train_loss = np.mean(train_loss)
            self.losses['train'].append(train_loss)

            # Validate
            if valloader is not None:
                if self.model_checkpoints.save_model_predictions:
                    val_loss, spearman_correlations, weighted_spearman, val_sequences, val_scores = self.evaluate(valloader)
                else:
                    val_loss, spearman_correlations, weighted_spearman = self.evaluate(valloader)
                self.losses['val'].append(val_loss)
                self.losses['corr'].append(weighted_spearman)
                # Update Spearman correlations
                for task, corr in spearman_correlations.items():
                    self.spearman[task].append(corr)
                predictions = {
                    'train': {'sequences': train_sequences, 'scores': train_scores},
                    'test': {'sequences': val_sequences, 'scores': val_scores}
                } if self.model_checkpoints.save_model_predictions else None
                self.model_checkpoints.update(self.model, -weighted_spearman, epoch, self.model_params, predictions)
                if weighted_spearman == float('nan'):
                    if self.verbose:
                        print(f"Stopping training because spearman correlations are NaNs.")
                    break

            # Optionally save models at specified intervals
            if epoch % self.save_interval == 0 and save_path is not None:
                print(f"Saving models to {save_path} on epoch {epoch}")
                self.model_checkpoints.save_models(save_path, current_epoch=epoch, filename=filename)

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if valloader is not None:
                        self.scheduler.step(val_loss)
                    else:
                        warnings.warn("ReduceLROnPlateau scheduler cannot be used when valloader is None.")
                else:
                    self.scheduler.step()

            # Print progress
            if self.verbose:
                text = f"Epoch {epoch}: train_Loss={train_loss:.4f}"
                if valloader is not None:
                    text += f", val_loss={val_loss:.4f}, corr={weighted_spearman:.4f}"
                text += f", l_rate={self.optimizer.param_groups[0]['lr']:.1e}"
                print(text)

            # Always keep track of the most recent completed epoch
            self.epoch = epoch

            # Use validation loss to initiate early stopping and determine best model
            if valloader is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                if no_improve_epochs >= stop_patience:
                    if self.verbose:
                        print(f"Stopping training because validation loss has not improved since {stop_patience} epochs")
                    if save_path is not None:
                        self.model_checkpoints.save_models(save_path, current_epoch=self.epoch, filename=filename)
                    break

        if save_path is not None:
            print(f"Saving models to {save_path} on epoch {epoch}")
            # save top models by model selection metric used
            self.model_checkpoints.save_models(save_path, current_epoch=epoch, filename=filename)
            # save the final model too
            predictions = {
                'train': {'sequences': train_sequences, 'scores': train_scores},
                'test': {'sequences': val_sequences, 'scores': val_scores}
            } if self.model_checkpoints.save_model_predictions else None
            self.model_checkpoints.save_final_model(self.model, self.model_params, save_path,
                                                    current_epoch=epoch, filename=filename,
                                                    val_loss=-self.losses['corr'][-1], predictions=predictions)

    def plot_losses(self, start_epoch=0, stop_epoch=None, savepath='training_losses.png', logscale=False):
        """
        Plots training and validation losses.

        Args:
            start_epoch (int, optional): The starting epoch for plotting. Default is 0.
            stop_epoch (int, optional): The stopping epoch for plotting. Default is None.
            savepath (str, optional): The path to save the plot. Default is 'training_losses.png'.
            logscale (bool, optional): Whether to use logarithmic scale for the y-axis. Default is False.

        Example:
            self.plot_losses(start_epoch=0, stop_epoch=100, savepath='losses.png', logscale=True)
        """
        if len(self.losses['train']) == 0:
            print('There are no losses in ModelTrainer.losses to plot. You have to train the model first')
            return
        losses = self.losses
        # Plot
        fig, ax1 = plt.subplots()
        epochs = range(1, len(losses['train']) + 1)
        ax1.plot(epochs[start_epoch:stop_epoch], losses['train'][start_epoch:stop_epoch], color='black', label='Train Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='black')
        if len(losses['val']) > 0:
            ax1.plot(epochs[start_epoch:stop_epoch], losses['val'][start_epoch:stop_epoch], color='brown', label='Val Loss')
        if logscale:
            ax1.set_yscale('log')
        if not logscale:
            ax1.set_ylim(0.0, 1.2)

        # To share a y-axis, we only use ax1 for plotting both train and val losses
        # Adding grid
        ax1.grid(True, which="both", linestyle='--', linewidth=0.5)

        # Adding legend
        ax1.legend()

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        plt.show(); plt.close()

    def plot_spearman(self, start_epoch=0, stop_epoch=None, savepath='spearman_correlations.png'):
        """
        Plots Spearman correlations for each task over epochs.

        Args:
            start_epoch (int, optional): The starting epoch for plotting. Default is 0.
            stop_epoch (int, optional): The stopping epoch for plotting. Default is None.
            savepath (str, optional): The path to save the plot. Default is 'spearman_correlations.png'.

        Example:
            self.plot_spearman(start_epoch=0, stop_epoch=100, savepath='spearman.png')
        """
        if not hasattr(self, 'spearman') or not self.spearman:
            print('There are no Spearman correlations in ModelTrainer.spearman to plot. You have to evaluate the model first')
            return

        # Plot
        fig, ax = plt.subplots()
        epochs = range(start_epoch + 1, stop_epoch + 1 if stop_epoch is not None else self.epoch + 1)
        for task, correlations in self.spearman.items():
            ax.plot(epochs, correlations[start_epoch:stop_epoch], label=task)

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Spearman Correlation')
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        ax.set_ylim(-0.51, 1.01)
        ax.legend()
        plt.title('Spearman Correlation Over Epochs by Task')
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()
        plt.close()


class ModelCheckpoint:
    """
    A class for managing and saving model checkpoints during training.

    Args:
        max_size (int): The maximum number of checkpoints to store. Default is 2.
        model_params (dict): Dictionary of model parameters.

    Attributes:
        max_size (int): Maximum number of checkpoints to store.
        models (list): List of stored model checkpoints, each as a tuple of (val_loss, epoch, model_state, model_params).
        predictions (list): List of stored predictions corresponding to the models.

    Example:
        checkpoint = ModelCheckpoint(max_size=3, model_params=model_params)
        checkpoint.update(model, val_loss, epoch, model_params, predictions)
        checkpoint.save_models(save_path, current_epoch, filename)
        checkpoint.save_final_model(model, model_params, save_path, current_epoch, val_loss, predictions, filename)
    """

    def __init__(self, max_size=2, model_params={}):
        self.max_size = max_size
        self.models = []
        self.predictions = []  # Save train and test sequences and model predictions, to facilitate testing inference code and model evaluation
        self.save_model_predictions = model_params.get('save_model_predictions', False)

    def update(self, model, val_loss, epoch, model_params, predictions=None):
        """
        Updates the list of model checkpoints with the current model state and predictions if save_model_predictions is True.

        Args:
            model (nn.Module): The current model.
            val_loss (float): The validation loss for the current epoch.
            epoch (int): The current epoch number.
            model_params (dict): Dictionary of model parameters.
            predictions (dict, optional): Dictionary containing train and test sequences and scores.

        Example:
            checkpoint.update(model, val_loss, epoch, model_params, predictions)
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model_state = self._get_non_esm_state_dict(model)

        # Append the new model and predictions
        self.models.append((val_loss, epoch, model_state, copy.deepcopy(model_params), model.__class__.__name__))
        if self.save_model_predictions:
            self.predictions.append(predictions)
        else:
            self.predictions.append(None)

        # Sort models and predictions based on val_loss
        combined = list(zip(self.models, self.predictions))
        combined.sort(key=lambda x: x[0][0])  # Sort by val_loss

        # Unzip the combined list back into models and predictions
        self.models, self.predictions = zip(*combined)

        # Keep only the top max_size models and predictions
        self.models = list(self.models[:self.max_size])
        self.predictions = list(self.predictions[:self.max_size])

        print(f"Updated model checkpoint - val_loss: {val_loss}, epoch: {epoch}")

    def save_models(self, save_path, current_epoch, filename=""):
        """
        Saves the current list of model checkpoints and their predictions to the specified path if save_model_predictions is True.

        Args:
            save_path (str): The path to save the checkpoints.
            current_epoch (int): The current epoch number.
            filename (str, optional): Prefix for the checkpoint filenames. Default is "".

        Example:
            checkpoint.save_models(save_path, current_epoch, filename)
        """
        for rank, ((val_loss, epoch, model_state, model_params, model_class_name),
                   prediction) in enumerate(zip(self.models, self.predictions), start=1):
            model_name = filename + f"_modelrank_{rank}_epoch_{epoch}_val_loss_{-val_loss:.4f}.pt"
            model_path = os.path.join(save_path, model_name)
            save_dict = {
                'model_state_dict': model_state,
                'model_params': model_params,
                'model_class_name': model_class_name
            }
            if self.save_model_predictions:
                save_dict['predictions'] = prediction
            torch.save(save_dict, model_path)

    def save_final_model(self, model, model_params, save_path, current_epoch, val_loss, predictions=None, filename=""):
        """
        Saves the final model and its predictions at the end of training if save_model_predictions is True.

        Args:
            model (nn.Module): The final model.
            model_params (dict): Dictionary of model parameters.
            save_path (str): The path to save the final model.
            current_epoch (int): The current epoch number.
            val_loss (float): The validation loss for the current epoch.
            predictions (dict, optional): Dictionary containing train and test sequences and scores.
            filename (str, optional): Prefix for the checkpoint filename. Default is "".

        Example:
            checkpoint.save_final_model(model, model_params, save_path, current_epoch, val_loss, predictions, filename)
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        final_model_path = os.path.join(save_path, f"{filename}_final_model_epoch_{current_epoch}_val_loss_{-val_loss:.4f}.pt")
        model_state = self._get_non_esm_state_dict(model)
        save_dict = {
            'model_state_dict': model_state,
            'model_params': model_params,
            'model_class_name': model.__class__.__name__
        }
        if self.save_model_predictions and predictions:
            save_dict['predictions'] = predictions
        torch.save(save_dict, final_model_path)
        print(f"Final model saved to {final_model_path}")

    def _get_non_esm_state_dict(self, model):
        """
        Removes the ESM model's parameters from the state dictionary.

        Args:
            model (nn.Module): The model to get the state dictionary from.

        Returns:
            dict: State dictionary without the ESM model's parameters.

        Example:
            state_dict = self._get_non_esm_state_dict(model)
        """
        # Remove DataParallel wrapper if present
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model_state = copy.deepcopy(model.state_dict())
        esm_prefix = "esm_model."
        non_esm_state = {k: v for k, v in model_state.items() if not k.startswith(esm_prefix)}
        return non_esm_state

    @staticmethod
    def load_model(model_path, model_class_lookup=None, print_num_parameters=True):
        """
        Loads a model from a checkpoint.

        Args:
            model_path (str): Path to the checkpoint file.
            model_class_lookup (dict): Dictionary mapping class names to model classes.
            print_num_parameters (bool, optional): Whether to print the number of model parameters. Default is True.

        Returns:
            tuple: Loaded model, model parameters, and predictions (if available).

        Example:
            model_class_lookup = {'ProteinFunctionPredictor': models.ProteinFunctionPredictor}
            model, model_params, predictions = ModelCheckpoint.load_model(model_path, model_class_lookup)
        """
        checkpoint = torch.load(model_path)
        model_params = checkpoint['model_params']

        # Dynamically load the model class
        model_class_name = checkpoint['model_class_name']
        if model_class_lookup is None:
            # Define the model class lookup dictionary
            model_class_lookup = {
                "ProteinFunctionPredictor_with_probmatrix": ProteinFunctionPredictor_with_probmatrix,
                #"ProteinFunctionPredictor": ProteinFunctionPredictor,
                # "GelmanCNN": GelmanCNN,
                # "VectorAttentionModel": VectorAttentionModel
            }
        if model_class_name not in model_class_lookup:
            raise ValueError(f"Model class '{model_class_name}' not found in model_class_lookup.")
        model_class = model_class_lookup[model_class_name]

        model = model_class(model_params)
        # Load model parameters
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

        if print_num_parameters:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters in the model: {total_params}")

        predictions = checkpoint.get('predictions', {})

        return model, model_params, predictions


def compute_spearman_correlation(predictions, targets):
    """
    Computes the Spearman correlation between predictions and targets.

    Args:
        predictions (list or np.array): Predicted values.
        targets (list or np.array): Target values.

    Returns:
        float: Spearman correlation coefficient.

    Example:
        corr = compute_spearman_correlation(predictions, targets)
    """
    # Convert lists to numpy arrays
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    valid_indices = ~np.isnan(predictions) & ~np.isnan(targets)
    if valid_indices.any():
        corr, _ = spearmanr(predictions[valid_indices], targets[valid_indices])
        return corr
    return np.nan
