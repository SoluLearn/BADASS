import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import random
import numpy as np
import pandas as pd
from badass.models.seq2fitness_models import ProteinFunctionPredictor_with_probmatrix
from badass.training.seq2fitness_traintools import ModelTrainer
from badass.data.datasets import ProteinDataset
from badass.utils.sequence_utils import split_by_sequence

def calculate_task_statistics(dataframe, task_columns):
    """
    Calculate the mean and standard deviation for the specified task columns in the dataframe.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        task_columns (list): List of column names for which to calculate statistics.

    Returns:
        tuple: A tuple containing:
            - task_means (dict): Dictionary of mean values for each task.
            - task_stds (dict): Dictionary of standard deviation values for each task.

    Example:
        task_means, task_stds = calculate_task_statistics(df, ['task1', 'task2'])
    """
    task_means = dataframe[task_columns].mean().to_dict()
    task_stds = dataframe[task_columns].std().to_dict()
    return task_means, task_stds

def load_data(dataframe, task_columns, batch_size=256, split_ratio=0.8, normalize_labels=True):
    """
    Load and split the data into training and validation sets, and create data loaders for each set.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        task_columns (list): List of column names for the tasks.
        batch_size (int, optional): Batch size for the data loaders. Default is 256.
        split_ratio (float, optional): Ratio to split the data into training and validation sets. Default is 0.8.
        normalize_labels (bool, optional): Whether to normalize the task labels. Default is True.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - task_means (dict): Dictionary of mean values for each task.
            - task_stds (dict): Dictionary of standard deviation values for each task.

    Example:
        train_loader, val_loader, task_means, task_stds = load_data(df, ['task1', 'task2'])
    """
    dataframe['sequence'].replace(np.nan, "NA", inplace=True) # Make wt NA for now.
    train_indices, val_indices = split_by_sequence(dataframe['sequence'].to_list(), split_ratio=split_ratio,
                                                              ref_seq="NA", return_sequences=False)
    dataframe['sequence'].replace("NA", "", inplace=True) # Now make WT empty string
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    train_dataframe = dataframe.iloc[train_indices]
    val_dataframe = dataframe.iloc[val_indices]
    print(f"After splitting, trainset has {len(train_dataframe)} sequences, and test has {len(val_dataframe)}.")

    # Calculate task statistics
    task_means, task_stds = calculate_task_statistics(train_dataframe, task_columns)

    train_dataset = ProteinDataset(train_dataframe, task_columns, task_means, task_stds, normalize_labels)
    val_dataset = ProteinDataset(val_dataframe, task_columns, task_means, task_stds, normalize_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, task_means, task_stds

def set_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generators.

    Example:
        set_seeds(42)
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

def main(model_params, training_params, model_class="ProteinFunctionPredictor_with_probmatrix"):
    """
    Initialize and train a model based on the provided parameters.

    Args:
        model_params (dict): Dictionary containing model parameters.
        training_params (dict): Dictionary containing training parameters.
        model_class (str, optional): The class name of the model to initialize. Default is "ProteinFunctionPredictor_with_probmatrix".

    Returns:
        ModelTrainer: The trained model trainer.

    Raises:
        ValueError: If an unknown model class is specified.

    Example:
        trainer = main(model_params, training_params, model_class="ProteinFunctionPredictor_with_probmatrix")
    """

    if model_class == "ProteinFunctionPredictor_with_probmatrix":
        print(f"Creating model of class ProteinFunctionPredictor_with_probmatrix.")
        model = ProteinFunctionPredictor_with_probmatrix(model_params)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    dropout = training_params.get('dropout', 0.2)
    model.reset_dropout(dropout)

    # Load data from a dataframe
    seed = training_params.get('seed', 7)
    set_seeds(seed)
    dataframe = pd.read_csv(training_params['dataset_path'], keep_default_na=False, na_values=[""])
    task_columns = list(model_params['task_criteria'].keys())
    ref_seq = model_params['ref_seq']

    normalize_labels = model_params.get('normalize_labels', True)
    train_loader, val_loader, task_means, task_stds = load_data(dataframe, task_columns,
                                                                batch_size=training_params.get('batch_size', 256),
                                                               normalize_labels = normalize_labels)
    if normalize_labels:
        task_stats = {'task_means': task_means, 'task_stds': task_stds}
        print(f"Task stats used for normalization are : {task_stats}.")
        model_params['task_stats'] = task_stats
        model.set_task_stats(model_params['task_stats'])

    # Get training parameters to simplify code later
    epochs = training_params.get('epochs', 100)
    learning_rate = training_params.get('lr', 5e-4)
    weight_decay = training_params.get('weight_decay', 1e-4)
    save_path = training_params.get('save_path', None)
    model_filename = training_params.get('model_filename', None)
    file_name = training_params.get('file_name', '')  # to save loss and correlation plots
    stop_patience = training_params.get('stop_patience', 1500)

    print(f"We'll train for {epochs} epochs.")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = ModelTrainer(model=model, model_params=model_params,
                           optimizer=optimizer, scheduler=scheduler,
                           verbose=True)
    trainer.train(train_loader, val_loader, max_epochs=epochs, save_path=save_path,
                  filename=model_filename, stop_patience=stop_patience)
    trainer.plot_losses(savepath=file_name + 'training_losses.png')
    trainer.plot_spearman(savepath=file_name + 'spearman_correlations.png')

    return trainer

if __name__ == "__main__":
    pass
