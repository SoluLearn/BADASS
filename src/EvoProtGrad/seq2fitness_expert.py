import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from evo_prot_grad.experts.base_experts import Expert
from evo_prot_grad.common.embeddings import OneHotEmbedding
from sequence_utils import rel_sequences_to_dict, create_absolute_sequences_list, convert_rel_seqs_to_tensors, pad_rel_seq_tensors_with_nan, list_of_dicts_from_list_of_sequence_strings
from seq2fitness_models import ProteinFunctionPredictor_with_probmatrix, compute_model_scores 
from seq2fitness_traintools import ModelCheckpoint


class CustomProteinExpert(Expert):
    def __init__(self, model_path: str, temperature: float, device: str = 'cpu', task_weights: Optional[Dict] = None):
        """
        Custom expert class for the ProteinFunctionPredictor_with_probmatrix model.
        Args:
            model_path (str): Path to the model checkpoint.
            temperature (float): Temperature for sampling from the expert.
            device (str): The device to use for the expert. Defaults to 'cpu'.
        """
        self.model, self.model_params, _ = ModelCheckpoint.load_model(model_path)
        self.tokenizer = self.model.alphabet.get_batch_converter()
        # Set requires_grad to True for ESM model parameters
        for param in self.model.esm_model.parameters():
            param.requires_grad = True
        # Apply OneHotEmbedding to the model's ESM embedding layer after loading the model
        self.model.esm_model.embed_tokens = OneHotEmbedding(self.model.esm_model.embed_tokens)
        self.model.esm_model.embed_tokens.weight = nn.Parameter(self.model.esm_model.embed_tokens.weight.half())
        self.model.eval()
        self.task_weights = task_weights
        
        self.temperature = temperature
        self.device = device
        self.model.to(device)
            
        vocab = {char: idx for idx, char in enumerate(self.model.alphabet.all_toks)}
        scoring_strategy = "dummy"  # Placeholder as we do not use variant_scoring
        super().__init__(temperature, self.model, vocab, scoring_strategy, device)
        
        self._wt_oh = None

    def _get_last_one_hots(self) -> torch.Tensor:
        """Returns the one-hot tensors most recently passed as input."""
        return self.model.esm_model.embed_tokens.one_hots

    def move_to_device(self, batch: Dict[str, Any], device: str) -> Dict[str, Any]:
        """Move batch to the specified device."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):  # In case of nested dictionaries
                batch[key] = {k: v.to(device) for k, v in value.items() if isinstance(v, torch.Tensor)}
        return batch

    def prepare_batch(self, inputs: List[str], device: str):
        """Prepare batch from absolute sequences """
        #print(f"inputs is {inputs}")
        _, rel_seqs_list_of_dicts = list_of_dicts_from_list_of_sequence_strings(inputs, self.model.ref_seq)
        #print(f"rel_seqs_list_of_dicts is {rel_seqs_list_of_dicts}.")
        batch_labels, batch_strs, batch_tokens = self.tokenizer([(str(i), seq) for i, seq in enumerate(inputs)])
        batch_tokens = batch_tokens.to(device)
        rel_seqs_tensors = convert_rel_seqs_to_tensors(rel_seqs_list_of_dicts)
        rel_seqs_tensors_padded = pad_rel_seq_tensors_with_nan(rel_seqs_tensors)
        return batch_tokens, rel_seqs_tensors_padded

    def tokenize(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes a list of protein sequences."""
        return self.prepare_batch(inputs, self.device)

    def get_model_output(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the one-hot sequences and model predictions for each amino acid in the input sequence."""
        inputs = [seq.replace(" ", "") for seq in inputs] # since EvoProtGrad adds spaces between amino acids, sigh
        batch_tokens, rel_seqs_tensors_padded = self.prepare_batch(inputs, self.device)
        self.move_to_device({'tokens': batch_tokens, 'rel_seqs': rel_seqs_tensors_padded}, self.device)
        predictions = self.model(batch_tokens, rel_seqs_tensors_padded)
        oh = self._get_last_one_hots()
        return oh, predictions

    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the one-hot sequences and expert score."""
        eps = 1e-40
        oh, predictions = self.get_model_output(inputs)
        scores = compute_model_scores(predictions, self.task_weights) / (self.temperature + eps)
        # scores = compute_model_scores(predictions, self.task_weights) 
        return oh.float(), scores.float()

    def init_wildtype(self, wt_seq: str) -> None:
        """Set the one-hot encoded wildtype sequence for this expert."""
        eps = 1e-40
        self._wt_oh, self._wt_preds = self.get_model_output([wt_seq])
        self._wt_score = compute_model_scores(self._wt_preds, self.task_weights).detach().cpu().numpy() / (self.temperature + eps)
        # self._wt_score = compute_model_scores(self._wt_preds, self.task_weights).detach().cpu().numpy()
        print(f"Wt sequence has score {float(self._wt_score):.4g}.")

def build_custom_expert(model_path: str, temperature: float, device: str = 'cpu', task_weights: Optional[Dict] = None) -> CustomProteinExpert:
    """Builds a CustomProteinExpert from a checkpoint."""
    return CustomProteinExpert(model_path=model_path, temperature=temperature, device=device, task_weights=task_weights)
    