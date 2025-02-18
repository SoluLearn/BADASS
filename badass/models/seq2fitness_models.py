import torch
from torch import nn
import torch.nn.functional as F
from badass.utils.sequence_utils import pseudolikelihood_ratio_from_tensor, compute_num_mutations_from_padded_rel_seqs_tensors
import os
import pandas as pd
import esm

eps = 1e-8
from badass.utils.sequence_utils import AMINO_ACIDS, AA_TO_IDX

class ProteinFunctionPredictor_with_probmatrix(nn.Module):
    """
    A neural network model for predicting protein function with optional relative embeddings and probability matrices.

    Args:
        model_params (dict): Dictionary containing model parameters including the reference sequence, model architecture settings, and task criteria.

    Attributes:
        ref_seq (str): The reference sequence of amino acids.
        use_rel_embeddings (bool): Whether to use relative embeddings.
        use_rel_prob_matrices (bool): Whether to use relative probability matrices.
        task_stats (dict): Dictionary containing task-specific statistics.
        criteria (dict): Dictionary containing task-specific loss functions and their weights.
        task_names (list): List of task names.
        k1 (int): Number of output channels for the first convolutional layer.
        k2 (int): Number of output channels for the second convolutional layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        quantiles (list): List of quantiles for computing statistics.
        m1 (int): Number of units in the first fully connected layer.
        m2 (int): Number of units in the second fully connected layer.
        esm_scores_dim (int): Dimension of the ESM scores.
        static_logit (tensor): Static logit tensor for computing ESM scores.
        esm_model_name (str): Name of the ESM model to use.
        device (torch.device): Device on which to run the model.

    Methods:
        reset_dropout(new_dropout): Reset the dropout rate.
        set_task_stats(task_stats): Set task-specific statistics.
        compute_stats(x, dim): Compute statistics (mean and quantiles) for the given tensor along the specified dimension.
        forward(batch_tokens, rel_seqs_tensors): Forward pass of the model.
    """
    def __init__(self, model_params):
        """
        Initialize the ProteinFunctionPredictor_with_probmatrix class.

        Args:
            model_params (dict): Dictionary containing model parameters including the reference sequence, model
            architecture settings, and task criteria.
        """
        super(ProteinFunctionPredictor_with_probmatrix, self).__init__()
        self._validate_model_params(model_params)
        self._initialize_model_params(model_params)
        self._initialize_esm_model()
        self._initialize_wt_embeddings()
        self._initialize_convolutions()
        self._initialize_fully_connected_layers()
        self._initialize_task_specific_layers()
        self._count_parameters()

    def _count_parameters(self):
        """
        Count and print the number of trainable parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        esm_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
        non_esm_params = total_params - esm_params
        print(f"Total number of trainable parameters in the model: {total_params}")
        print(f"Number of trainable parameters in the model excluding ESM: {non_esm_params}")

    def _validate_model_params(self, model_params):
        """
        Validate the model parameters to ensure the reference sequence is provided.

        Args:
            model_params (dict): Dictionary containing model parameters.

        Raises:
            ValueError: If the reference sequence is not provided in model_params.
        """
        if 'ref_seq' not in model_params or not isinstance(model_params['ref_seq'], str):
            raise ValueError("model_params must have the reference sequence of amino acids as a string.")

    def _initialize_model_params(self, model_params):
        """
        Initialize model parameters from the provided dictionary.

        Args:
            model_params (dict): Dictionary containing model parameters.
        """
        self.ref_seq = model_params['ref_seq']
        self.use_rel_embeddings = model_params.get('use_rel_embeddings', True)
        self.use_rel_prob_matrices = model_params.get('use_rel_prob_matrices', False)
        self.task_stats = model_params.get('task_stats', {})
        self.criteria = model_params.get('task_criteria', {'main': {'loss': nn.MSELoss(), 'weight': 1.0}})
        self.task_names = list(self.criteria.keys())
        self.k1 = model_params.get('k1', 64)
        self.k2 = model_params.get('k2', 64)
        self.dropout = nn.Dropout(model_params.get('dropout', 0.20))
        self.quantiles = model_params.get('quantiles', [0.025, 0.25, 0.5, 0.75, 0.975])
        self.m1 = model_params.get('m1', 18)
        self.m2 = model_params.get('m2', 9)
        self.esm_scores_dim = model_params.get('esm_scores_dim', 3)
        self.static_logit = model_params['static_logit']
        self.esm_model_name = model_params.get('esm_modelname', 'esm2_t33_650M_UR50D')
        if torch.cuda.is_available():
            self.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else:
            self.devices = [torch.device('cpu')]
        self.device = self.devices[0]

    def _initialize_esm_model(self):
        """
        Initialize the ESM model for feature extraction.
        """
        self.esm_model, self.alphabet = eval(f'esm.pretrained.{self.esm_model_name}()')
        self.aa_indices = [self.alphabet.get_idx(aa) for aa in AMINO_ACIDS]
        self.num_layers = self.esm_model.num_layers
        self.esm_h_dim = self.esm_model.embed_dim
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.esm_model.to(self.device)
        self.esm_model.eval().half()

    def _initialize_wt_embeddings(self):
        """
        Initialize the wild-type embeddings and log probabilities using the ESM model.
        """
        batch_converter = self.alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter([("", self.ref_seq)])
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model(tokens=batch_tokens, repr_layers=[self.num_layers], return_contacts=False)
            self.logprob_wt = nn.functional.log_softmax(results['logits'][:, 1:-1, :], dim=-1)[:, :, self.aa_indices].to(self.device)
            self.embedding_wt = results['representations'][self.num_layers][:, 1:-1, :].float().to(self.device)
        del batch_converter

    def _initialize_convolutions(self):
        """
        Initialize convolutional layers for processing embeddings and probability matrices.
        """
        self.conv1 = nn.Conv1d(in_channels=self.esm_h_dim, out_channels=self.k1, kernel_size=1).to(self.device)
        self.conv2 = nn.Conv1d(in_channels=self.esm_h_dim, out_channels=self.k2, kernel_size=1).to(self.device)
        self.prob_conv1 = nn.Conv1d(in_channels=20, out_channels=self.k1, kernel_size=1).to(self.device)
        self.prob_conv2 = nn.Conv1d(in_channels=20, out_channels=self.k2, kernel_size=1).to(self.device)

    def _initialize_fully_connected_layers(self):
        """
        Initialize fully connected layers for processing the combined features.
        """
        n_stats = len(self.quantiles) + 1
        self.fc1 = nn.Linear((self.k1 + self.k2) * n_stats * 2 + 2 * self.esm_scores_dim + 1, self.m1).to(self.device)
        self.fc2 = nn.Linear(self.m1, self.m2).to(self.device)

    def _initialize_task_specific_layers(self):
        """
        Initialize task-specific output layers and biases.
        """
        self.task_specific_layers = nn.ModuleDict()
        self.biases = nn.ParameterDict()
        for task in self.task_names:
            self.task_specific_layers[task] = nn.Linear(self.m2, 1).to(self.device)
            self.biases[task] = nn.Parameter(torch.zeros(1)).to(self.device)

    def reset_dropout(self, new_dropout):
        """
        Reset the dropout rate to a new value.

        Args:
            new_dropout (float): New dropout rate.
        """
        self.dropout = nn.Dropout(new_dropout)

    def set_task_stats(self, task_stats):
        """
        Set task-specific statistics.

        Args:
            task_stats (dict): Dictionary containing task-specific statistics.
        """
        self.task_stats = task_stats

    def compute_stats(self, x, dim):
        """
        Compute statistics (mean and quantiles) for the given tensor along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): Dimension along which to compute statistics.

        Returns:
            torch.Tensor: Tensor containing the computed statistics.
        """
        quantiles = torch.tensor(self.quantiles, device=x.device)
        mean = torch.nanmean(x, dim=dim, keepdim=True).permute(0, 2, 1)
        q = torch.quantile(x, quantiles, dim=dim, keepdim=True).squeeze(dim + 1).permute(1, 2, 0)
        stats = torch.cat((mean, q), dim=-1)
        return stats

    def forward(self, batch_tokens, rel_seqs_tensors):
        """
        Perform the forward pass of the model.

        Args:
            batch_tokens (torch.Tensor): Tokenized batch of sequences.
            rel_seqs_tensors (torch.Tensor): Relative sequence tensors.

        Returns:
            dict: Dictionary containing task-specific outputs.
        """
        num_mutations = compute_num_mutations_from_padded_rel_seqs_tensors(rel_seqs_tensors).to(batch_tokens.device)
        embeddings, logprobs = self._forward_esm_model(batch_tokens)
        return self._compute_forward_outputs(embeddings, logprobs, num_mutations, rel_seqs_tensors)

    def _forward_esm_model(self, inputs):
        """
        Forward pass through the ESM model.

        Args:
            inputs (torch.Tensor): Input tensor for the ESM model.

        Returns:
            tuple: Embeddings and log probabilities from the ESM model.
        """
        results = self.esm_model(tokens=inputs, repr_layers=[self.num_layers], return_contacts=False)
        embeddings = results['representations'][self.num_layers][:, 1:-1, :].float()
        logprobs = nn.functional.log_softmax(results['logits'][:, 1:-1, :], dim=-1)[:, :, self.aa_indices].float()
        return embeddings, logprobs

    def _compute_forward_outputs(self, embeddings, logprobs, num_mutations, rel_seqs_tensors):
        """
        Compute the forward outputs of the model.

        Args:
            embeddings (torch.Tensor): Embeddings from the ESM model.
            logprobs (torch.Tensor): Log probabilities from the ESM model.
            num_mutations (torch.Tensor): Tensor containing the number of mutations for each sequence.
            rel_seqs_tensors (torch.Tensor): Relative sequence tensors.

        Returns:
            dict: Dictionary containing task-specific outputs.
        """
        if self.use_rel_embeddings:
            embeddings -= self.embedding_wt.to(embeddings.device)
        stats_path1 = self.compute_stats(embeddings, dim=1)
        x_conv1 = self.dropout(self.conv1(stats_path1)).view(stats_path1.size(0), -1)
        x_permuted_for_conv2 = embeddings.permute(0, 2, 1)
        x_conv2_raw = self.dropout(self.conv2(x_permuted_for_conv2))
        x_conv2_raw_permuted = x_conv2_raw.permute(0, 2, 1)
        stats_path2 = self.compute_stats(x_conv2_raw_permuted, dim=1).view(x_conv2_raw.size(0), -1)

        if self.use_rel_prob_matrices:
            logprobs -= self.logprob_wt.to(logprobs.device)
        prob_stats = self.compute_stats(logprobs, dim=1)
        prob_conv1 = self.dropout(self.prob_conv1(prob_stats)).view(prob_stats.size(0), -1)
        prob_permuted_for_conv2 = logprobs.permute(0, 2, 1)
        prob_conv2_raw = self.dropout(self.prob_conv2(prob_permuted_for_conv2))
        prob_conv2_raw_permuted = prob_conv2_raw.permute(0, 2, 1)
        prob_stats_path2 = self.compute_stats(prob_conv2_raw_permuted, dim=1).view(prob_conv2_raw.size(0), -1)

        esm_scores = self._compute_esm_scores(logprobs, rel_seqs_tensors)
        combined_features = torch.cat([x_conv1, stats_path2, prob_conv1, prob_stats_path2,
                                       num_mutations, esm_scores, esm_scores / (num_mutations + eps)], dim=1)

        x_fc1 = F.gelu(self.dropout(self.fc1(combined_features)))
        x_fc2 = F.gelu(self.fc2(x_fc1))
        return self._compute_task_outputs(x_fc2)

    def _compute_esm_scores(self, logprobs, rel_seqs_tensors):
        """
        Compute ESM scores for the given sequences.

        Args:
            logprobs (torch.Tensor): Log probabilities from the ESM model.
            rel_seqs_tensors (torch.Tensor): Relative sequence tensors.

        Returns:
            torch.Tensor: Tensor containing the computed ESM scores.
        """
        esm_scores_mutant, esm_scores_ref, esm_scores_3B = [], [], []
        for seq_idx, rel_seq_tensor in enumerate(rel_seqs_tensors):
            esm_scores_mutant.append(pseudolikelihood_ratio_from_tensor(rel_seq_tensor, self.ref_seq, logprobs[seq_idx].squeeze()))
            esm_scores_ref.append(pseudolikelihood_ratio_from_tensor(rel_seq_tensor, self.ref_seq, self.logprob_wt.squeeze()))
            esm_scores_3B.append(pseudolikelihood_ratio_from_tensor(rel_seq_tensor, self.ref_seq, self.static_logit))
        esm_scores_mutant = torch.tensor(esm_scores_mutant, dtype=torch.float32).to(logprobs.device)
        esm_scores_ref = torch.tensor(esm_scores_ref, dtype=torch.float32).to(logprobs.device)
        esm_scores_3B = torch.tensor(esm_scores_3B, dtype=torch.float32).to(logprobs.device)
        return torch.stack([esm_scores_mutant, esm_scores_ref, esm_scores_3B], dim=1)

    def _compute_task_outputs(self, x_fc2):
        """
        Compute the task-specific outputs.

        Args:
            x_fc2 (torch.Tensor): Output from the second fully connected layer.

        Returns:
            dict: Dictionary containing task-specific outputs.
        """
        output_dict = {}
        for task_name in self.task_names:
            output_dict[task_name] = self.task_specific_layers[task_name](x_fc2) + self.biases[task_name]
        return output_dict

def initialize_static_esm_scores(file_path, verbose=False):
    """
    Load and initialize static ESM scores from an Excel file.

    Args:
        file_path (str): Path to the Excel file containing the static single mutant scores.
        verbose (bool, optional): Whether to print verbose output. Default is False.

    Returns:
        torch.Tensor: Tensor containing the loaded static ESM scores.

    Example:
        logit_tensor = initialize_static_esm_scores('path/to/esm_scores.xlsx', verbose=True)
    """
    if verbose:
        print(f"Loading static single mutant scores from path {file_path}.")
    matrix_data = pd.read_excel(file_path, sheet_name='matrix')
    logit_matrix = matrix_data.iloc[:, 2:]
    logit_matrix = logit_matrix[list(AMINO_ACIDS)]
    logit_tensor = torch.tensor(logit_matrix.values, dtype=torch.float32)
    return logit_tensor

def unnormalize_predictions(predictions, task_means, task_stds):
    """
    Unnormalize the predicted values using the provided means and standard deviations for each task.

    Args:
        predictions (dict): Dictionary of normalized predictions for each task.
        task_means (dict): Dictionary of mean values for each task.
        task_stds (dict): Dictionary of standard deviation values for each task.

    Returns:
        dict: Dictionary of unnormalized predictions for each task.

    Example:
        unnormalized_preds = unnormalize_predictions(predictions, task_means, task_stds)
    """
    unnormalized_predictions = {task: (predictions[task] * task_stds[task]) + task_means[task] for task in predictions}
    return unnormalized_predictions

def compute_model_scores(predictions, task_prediction_weights = None):
        """
        Computes weighted prediction scores based on task weights.

        Args:
            predictions (dict): Dictionary of predictions for each task.
            task_prediction_weights (dict): Dictionary with the names and weights to use to combine task predictions. If None,
            use all tasks in predictions with equal weights.

        Returns:
            numpy.ndarray: Weighted prediction scores.

        Example:
            pred_scores = compute_prediction_scores(predictions, task_prediction_weights)
        """
        pred_scores = None
        sum_w = 0.0
        if task_prediction_weights is None:
            task_prediction_weights = {key: 1.0 for key in predictions}
        for task in task_prediction_weights.keys():
            w = task_prediction_weights[task]
            if w > 0:
                if pred_scores is None:
                    pred_scores = w * predictions[task]
                else:
                    pred_scores += w * predictions[task]
                sum_w += w
        return pred_scores / (sum_w + eps)
