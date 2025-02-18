from torch.utils.data import Dataset
import torch
from torch import tensor, stack, cat, float16, long

eps = 1e-8

class ProteinDataset(Dataset):
    def __init__(self, dataframe, task_columns, task_means, task_stds, normalize_labels=True):
        """
        A dataset class with sequences as strings and measurements as task labels.
        
        Args:
            dataframe (pd.DataFrame): Dataframe containing the sequences and task labels.
            task_columns (list): List of column names for the task labels.
            task_means (dict): Dictionary of means for each task.
            task_stds (dict): Dictionary of standard deviations for each task.
            normalize_labels (bool): Whether to normalize labels.
        """
        # Filter out rows with NaN values in all task columns
        self.dataframe = dataframe.dropna(subset=task_columns, how='all')
        self.sequences = self.dataframe['sequence'].tolist()
        #print(f"The {len(self.sequences)} sequences in this dataset are {self.sequences}.")
        self.task_labels = self.dataframe[task_columns].values
        self.task_columns = task_columns
        self.task_means = task_means
        self.task_stds = task_stds
        self.normalize_labels = normalize_labels

        # Precompute labels
        self.labels_list = []
        num_all_nan = 0
        for idx in range(len(self.sequences)):
            labels = {task: (self.task_labels[idx][i] - self.task_means[task]) / (self.task_stds[task] + 1e-8) 
                      for i, task in enumerate(self.task_columns)} if self.normalize_labels else \
                     {task: self.task_labels[idx][i] 
                      for i, task in enumerate(self.task_columns)}
            if all(torch.isnan(torch.tensor(list(labels.values())))):
                num_all_nan += 1
            self.labels_list.append(labels)

        print(f"Number of data points with all NaNs after normalization: {num_all_nan}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sequence = self.sequences[idx]
        labels = {task: torch.tensor(self.labels_list[idx][task], dtype=torch.float32) for task in self.task_columns}
        return {'sequence': sequence, 'labels': labels}


class SequenceEmbeddingDataset(Dataset):
    """
    A dataset class for sequence embeddings, supporting supervised, unsupervised,
    semi-supervised, and multi-task learning scenarios. Assumes all embeddings are
    torch tensors of the same size. It now also supports storing and handling 
    probability matrices for each sequence.

    Attributes:
        seq_names (list of str): Names or identifiers for each sequence.
        embeddings (torch.Tensor): Tensor containing all sequence embeddings.
        num_mutations (torch.Tensor): Tensor containing the number of mutations per sequence.
        reference_sequence (str): The reference sequence for comparison.
        labels (list of torch.Tensor or None): Labels for each sequence, if applicable.
        label_names (list of str or None): Names of the labels for multi-task learning.
        fft_mag (torch.Tensor or None): Tensor containing FFT magnitude for each sequence, if applicable.
        fft_phase (torch.Tensor or None): Tensor containing FFT phase for each sequence, if applicable.
        esm_scores (torch.Tensor or None): Tensor containing ESM scores for each sequence, potentially multi-dimensional.
        prob_matrices (torch.Tensor or None): Tensor stack of probability matrices for each sequence, if provided.

    Methods:
        __init__: Initializes the dataset with sequence data and optional labels, FFT data, scores, and probability matrices.
        __len__: Returns the number of sequences in the dataset.
        __getitem__: Retrieves a dictionary with all data attributes for a sequence by index.
        merge: Merges another dataset of the same structure into this one, extending all attributes.
    """
    def __init__(self, seq_names, embeddings, num_mutations, ref_seq, labels=None, label_names=None,
                 fft_mag=None, fft_phase=None, esm_scores=None, prob_matrices=None):
        assert len(seq_names) == len(embeddings) == len(num_mutations), "All inputs must have the same length."
        if labels is not None:
            assert len(labels) == len(seq_names), "Labels must match the number of sequences."
            assert label_names is not None, "Label names must be provided with labels."
        if fft_mag is not None:
            assert len(fft_mag) == len(seq_names) == len(fft_phase), "FFT coefficients must match the number of sequences."
        if prob_matrices is not None:
            assert len(prob_matrices) == len(seq_names), "Probability matrices must match the number of sequences."

        self.seq_names = seq_names
        self.embeddings = torch.stack(embeddings)
        self.num_mutations = torch.tensor(num_mutations, dtype=torch.long)
        self.reference_sequence = ref_seq
        self.labels = [torch.tensor(label) for label in labels] if labels is not None else None
        self.label_names = label_names
        self.fft_mag = torch.stack(fft_mag) if fft_mag else None
        self.fft_phase = torch.stack(fft_phase) if fft_phase else None
        self.esm_scores = torch.tensor(esm_scores, dtype=torch.float16) if esm_scores is not None else None
        self.prob_matrices = torch.stack(prob_matrices) if prob_matrices is not None else None

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        item = {
            'seq_name': self.seq_names[idx],
            'embedding': self.embeddings[idx],
            'num_mutations': self.num_mutations[idx],
            'label': self.labels[idx] if self.labels is not None else None,
            'esm_score': self.esm_scores[idx] if self.esm_scores is not None else None,
        }
        # Check if fft_mag and fft_phase are not None before trying to index
        if self.fft_mag is not None:
            item['fft_mag'] = self.fft_mag[idx]
        else:
            item['fft_mag'] = None

        if self.fft_phase is not None:
            item['fft_phase'] = self.fft_phase[idx]
        else:
            item['fft_phase'] = None

        # Also check prob_matrices
        if self.prob_matrices is not None:
            item['prob_matrix'] = self.prob_matrices[idx]
        else:
            item['prob_matrix'] = None

        return item
