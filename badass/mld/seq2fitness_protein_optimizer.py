import torch
import random
import copy
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import esm
from typing import List
from io import StringIO
from badass.training.seq2fitness_traintools import ModelCheckpoint
from badass.models.seq2fitness_models import compute_model_scores
from badass.mld.sequence_sampler import CombinatorialMutationSampler
from badass.utils.sequence_utils import (AMINO_ACIDS,
                    AA_TO_IDX,
                    EPS,
                    rel_sequences_to_dict,
                    convert_rel_seqs_to_tensors,
                    pad_rel_seq_tensors_with_nan,
                    apply_mutations,
                    pseudolikelihood_ratio_from_tensor,
                    generate_single_mutants,
                    compute_Neff_for_probmatrix,
                    create_score_matrix)

## To study speed up opportunities only
import cProfile
import pstats

class ProteinOptimizer:
    """
    Optimize protein sequences using simulated annealing with an optional phase transition detection mechanism.

    Args:
        optimizer_params (dict): Dictionary containing parameters for the optimization process.
        model_params (dict, optional): Dictionary containing model parameters for ESM-only inference. Default is None.
        model_checkpoint_path (str, optional): Path to the model checkpoint for full ML model inference. Default is None.
        esm_only (bool, optional): Whether to use only ESM scores without an ML model. Default is False.
        score_matrix (np.ndarray, optional): Initial score matrix. Default is None, leading to all single mutant score computations.
        simple_simulated_annealing (bool, optional): Whether to use simple simulated annealing without adaptation. Default is False.
        cool_then_heat (bool, optional): Whether to use a cooling-then-heating schedule. Default is False.

    Example:
        model_path = 'model_checkpoint.pt'
        optimizer_params = {
            'seqs_per_iter': 520,
            'num_iter': 100,
            'init_score_batch_size': 500,
            'T': 2.0,
            'seed': 7,
            'gamma': 0.0,
            'cooling_rate': 0.9,
            'num_mutations': 5,
            'sites_to_ignore': [1, 330]
        }
        # load initial score matrix if available, otherwise set to None or omit argument
        with open('single_mutant_scores.pkl', 'rb') as file:
            initial_score_matrix = pickle.load(file)
        optimizer = ProteinOptimizer(optimizer_params, model_checkpoint_path=model_path, model_params=None, esm_only=False,
                                     score_matrix=initial_score_matrix, simple_simulated_annealing=False, cool_then_heat=False)
        df, df_stats = optimizer.optimize()
        optimizer.plot_scores()
        optimizer.save_results(filename='results.csv', n_seqs_to_keep=10000)
    """
    def __init__(self, optimizer_params, model_params=None, model_checkpoint_path=None, esm_only=False,
                 score_matrix=None, simple_simulated_annealing=False, cool_then_heat=False):
        """
        Initialize the ProteinOptimizer class.

        Args:
            optimizer_params (dict): Dictionary containing parameters for the optimization process.
            model_params (dict, optional): Dictionary containing model parameters for ESM-only inference. Default is None.
            model_checkpoint_path (str, optional): Path to the model checkpoint for full ML model inference. Default is None.
            esm_only (bool, optional): Whether to use only ESM scores without the ML model. Default is False.
            score_matrix (np.ndarray, optional): Initial score matrix. Default is None.
            simple_simulated_annealing (bool, optional): Whether to use simple simulated annealing without adaptation. Default is False.
            cool_then_heat (bool, optional): Whether to use a cooling-then-heating schedule. Default is False.
            normalize_scores (bool, optional): Whether to normalize sequence scores before computing negative energies for sampler. Can
            provide numerical stability, and lead to an initial temperature of about 1.0.
        """
        self.optimizer_params = optimizer_params
        self.esm_only = esm_only
        self.model_checkpoint_path = model_checkpoint_path
        self._setup_optimizer_params()
        self._setup_model(model_params)
        self._set_seeds()
        if self.esm_only:
            self.wt_score = np.array(0.0, dtype=np.float32) # by definition
        else:
            self.wt_score = self._get_wildtype_score()
            self._set_ml_model_score_thresholds()
        print(f"The reference sequence has score {float(self.wt_score):.3g}.")
        self.sampler= self._initialize_sampler(score_matrix)
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        gc.collect()

    def _setup_optimizer_params(self):
        """
        Set up optimizer parameters from the optimizer_params dictionary.
        """
        self.normalize_scores = self.optimizer_params.get('normalize_scores', True) # subtract reference value and divide by scale
        self.T = self.optimizer_params.get('T', 1.5)
        self.n_batch = self.optimizer_params.get('seqs_per_iter', 500)
        self.init_score_batch_size = self.optimizer_params.get('init_score_batch_size', 500)
        self.total_iter = self.optimizer_params.get('num_iter', 200)
        self.n_seqs = self.total_iter * self.n_batch
        self.seed = self.optimizer_params.get('seed', 7)
        self.gamma = self.optimizer_params.get('gamma', 0.5)
        self.boost_mutations_with_high_variance = self.optimizer_params.get('boost_mutations_with_high_variance', True)
        self.sample_variety_of_mutation_numbers = self.optimizer_params.get('sample_variety_of_mutation_numbers', False)
        self.num_sequences_proportions = self.optimizer_params.get('num_sequences_proportions', [0, 1, 1, 1, 1, 1])
        self.cooling_rate = self.optimizer_params.get('cooling_rate', 0.92)
        self.num_mutations = self.optimizer_params.get('num_mutations', 5)
        self.n_seqs_to_keep = self.optimizer_params.get('n_seqs_to_keep', None) # if None keep all sequences
        self.sites_to_ignore = self.optimizer_params.get('sites_to_ignore', [1])
        self.simple_simulated_annealing = self.optimizer_params.get('simple_simulated_annealing', False)
        self.cool_then_heat = self.optimizer_params.get('cool_then_heat', False)

        self.all_sequences, self.all_scores = [], []
        self.iters_for_seqs = []
        self.n_eff_joint, self.n_eff_sites, self.n_eff_aa, self.all_Ts = [], [], [], []
        self.seq_iter_counts = []
        self.all_variances = []
        self.all_num_new_seqs = []
        self.all_num_sampled_seqs = []
        self.all_mean_ml_scores = []
        self.all_phase_transition_numbers = []
        self.scored_sequences = {} # relative sequence to scores
        self.mutation_score_map = {} # mutation to list of scores

        self.active_phase_transition = False
        self.first_phase_transition, self.initial_var, self.initial_ml_score = None, None, None
        self.last_high_ml_score = float('-inf')
        self.last_high_var = float('-inf')
        self.last_phase_transition = None
        self.num_phase_transitions = 0
        self.patience_phase_trans = 10 if self.esm_only else 3
        self.patience = 10 if (self.num_mutations > 3) else 15
        self.patience += self.patience_phase_trans
        self.heating_rate = 1.4 if (self.num_mutations > 3) else 1.6
        self.heating_rate = 1.8 if self.esm_only else self.heating_rate
        self.low_temp_threshold = 0.02
        self.high_temp_threshold = 6 if self.esm_only else 1.6 # for cool then heat runs

    def _setup_model(self, model_params):
        """
        Set up the model for inference based on the provided model parameters.

        Args:
            model_params (dict): Dictionary containing model parameters for ESM-only inference.
        """
        self.num_gpus = torch.cuda.device_count()  # Needed to handle bypassing DataParallel for small batches
        if torch.cuda.is_available():
            self.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else:
            self.devices = [torch.device('cpu')]
        self.device = self.devices[0]
        print(f"Our main device will be {self.device}")
        if self.esm_only:
            if model_params is None:
                raise ValueError("model_params must be provided for ESM only inference with model name, ref seq and (optionally) weights.")
            self.ref_seq = model_params.get('ref_seq')
            self.esm_model_name = model_params.get('esm_modelname', 'esm2_t33_650M_UR50D')
            self.esm_score_weights = model_params.get('esm_score_weights', None)
            self._initialize_esm_model(model_params)
        else:
            self.model, self.model_params, _ = ModelCheckpoint.load_model(self.model_checkpoint_path)
            self.tokenizer = self.model.alphabet.get_batch_converter() # Esm tokenizer since fwd method expects tokens
            self.ref_seq = self.model_params.get('ref_seq')
            task_weights=self.optimizer_params.get('new_task_weights', None)
            if task_weights is None:
                task_weights = {task: self.model_params['task_criteria'][task]['weight'] for task in self.model_params['task_criteria'].keys()}
            self.task_weights = task_weights
        print(f"The reference sequence is {self.ref_seq}.")
        # Wrap the model in DataParallel
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            print(f"We'll use {self.num_gpus} GPUs through DataParallel.")
        self.model.eval().to(self.device)  # Move the model to the primary device first
        self.device = next(self.model.parameters()).device
        print(f"Our main device is still {self.device}")
        if self.esm_only:
            # Get reference logits to compute score
            batch_tokens, rel_seqs_tensors_padded = self.prepare_batch([''], self.device)
            with torch.no_grad():
                if isinstance(self.model, torch.nn.DataParallel):
                    esm_output = self.model.module(tokens=batch_tokens)
                else:
                    esm_output = self.model(tokens=batch_tokens)
                self.wt_logits = esm_output['logits'][:, 1:-1, self.aa_indices].squeeze() # this is tensor([1, L, 20])

    def _set_ml_model_score_thresholds(self, k_low=0.0):
        """
        Set thresholds for ML model score optimization to detect phase transitions and reversals.

        Args:
            k_low (float, optional): Threshold for detecting a phase transition reversal. Default is 0.0.
        """
        self.score_threshold = self.optimizer_params.get('score_threshold', None) # to detect phase transition when using ESM only scores
        if self.score_threshold is None:
            f = 1 + 0.01 * (self.num_mutations - 2) if self.num_mutations >= 2 else 1
            self.score_threshold = f * self.wt_score
        self.reversal_threshold = self.optimizer_params.get('reversal_threshold', None) # to detect phase transition reversal
        if self.reversal_threshold is None:
            self.reversal_threshold = k_low

    def _initialize_esm_model(self, model_params):
        """
        Initialize the ESM model for inference.

        Args:
            model_params (dict): Dictionary containing model parameters for ESM-only inference.
        """
        self.model, self.alphabet = eval(f'esm.pretrained.{self.esm_model_name}()')
        self.aa_indices = [self.alphabet.get_idx(aa) for aa in AMINO_ACIDS]
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.half()
        self.tokenizer = self.model.alphabet.get_batch_converter() # For ESM model
        self._set_esm_score_thresholds()
        self.esm_score_weights = model_params.get('esm_score_weights', {'mutant': 0.5, 'reference': 0.5})
        # normalize weights
        sum_w = self.esm_score_weights['mutant'] + self.esm_score_weights['reference']
        self.esm_score_weights['mutant'] /= sum_w
        self.esm_score_weights['reference'] /= sum_w

    def _set_esm_score_thresholds(self, k_mutant = 20.0, k_ref=11.0, k_mutant_rev=0.0, k_ref_rev=-10.0):
        """
        Set thresholds for ESM score optimization to detect phase transitions and reversals.

        Args:
            k_mutant (float, optional): Threshold for ESM mutant sequence scores. Default is 11.9.
            k_ref (float, optional): Threshold for ESM reference sequence scores. Default is 5.0.
            k_mutant_rev (float, optional): Threshold for ESM mutant sequence score phase reversal. Default is 0.0.
            k_ref_rev (float, optional): Threshold for ESM reference sequence score phase reversal. Default is -10.0.
        """
        self.score_threshold = self.optimizer_params.get('score_threshold', None) # to detect phase transition when using ESM only scores
        if self.score_threshold is None:
            mutant_weight = self.esm_score_weights['mutant']
            reference_weight = self.esm_score_weights['reference']
            self.score_threshold = mutant_weight*k_mutant + reference_weight*k_ref
        self.reversal_threshold = self.optimizer_params.get('reversal_threshold', None) # to detect phase transition reversal
        if self.reversal_threshold is None:
            mutant_weight = self.esm_score_weights['mutant']
            reference_weight = self.esm_score_weights['reference']
            self.reversal_threshold = k_mutant_rev*mutant_weight + k_ref_rev*reference_weight

    def _get_wildtype_score(self):
        """
        Compute the score for the reference (wild-type) sequence. Only relevant when self.esm_only is False.

        Returns:
            float: Score of the reference sequence.
        """
        batch_tokens, rel_seqs_tensors_padded = self.prepare_batch([''], self.device)
        with torch.no_grad():
            if isinstance(self.model, torch.nn.DataParallel):
                predictions = self.model.module(batch_tokens, rel_seqs_tensors_padded)  # Bypass DataParallel since just one seq
            else:
                predictions = self.model(batch_tokens, rel_seqs_tensors_padded)
            wt_score = compute_model_scores(predictions, self.task_weights).squeeze()
        return wt_score.detach().cpu().numpy()

    def prepare_batch(self, inputs: List[str], device: str):
        """
        Prepare a batch of sequences for model inference.

        Args:
            inputs (List[str]): List of relative sequences as strings.
            device (str): Device to run the inference on.

        Returns:
            tuple: Tokenized batch of sequences and padded relative sequence tensors.
        """
        # Need rel_seqs_list_of_dicts and list of absolute strings
        rel_seqs_dict = rel_sequences_to_dict(inputs, sep='-') # nested dictionary maintaining order...
        rel_seqs_list_of_dicts = [rel_seqs_dict[key] for key in sorted(rel_seqs_dict)] # maintains order because we added the index
        abs_seqs = [apply_mutations(self.ref_seq, rel_seq) for rel_seq in rel_seqs_list_of_dicts]
        batch_labels, batch_strs, batch_tokens = self.tokenizer([(str(i), seq) for i, seq in enumerate(abs_seqs)])
        rel_seqs_tensors = convert_rel_seqs_to_tensors(rel_seqs_list_of_dicts)
        rel_seqs_tensors_padded = pad_rel_seq_tensors_with_nan(rel_seqs_tensors)
        return batch_tokens.to(device), rel_seqs_tensors_padded.to(device)

    def _compute_esm_scores(self, batch_tokens, rel_seqs_tensors_padded):
        """
        Compute ESM scores for the given batch of sequences.

        Args:
            batch_tokens (torch.Tensor): Tokenized batch of sequences.
            rel_seqs_tensors_padded (torch.Tensor): Padded relative sequence tensors.

        Returns:
            np.ndarray: Computed ESM scores for the batch.
        """
        with torch.no_grad():
            mutant_logits = self.model(tokens=batch_tokens)['logits'][:, 1:-1, self.aa_indices]  # tensor([batch, L, 20])

        mutant_scores, ref_scores = [], []
        for seq_idx, rel_seq_tensor in enumerate(rel_seqs_tensors_padded):
            mut_score = pseudolikelihood_ratio_from_tensor(rel_seq_tensor, self.ref_seq, mutant_logits[seq_idx, :, :].squeeze())
            ref_score = pseudolikelihood_ratio_from_tensor(rel_seq_tensor, self.ref_seq, self.wt_logits)
            mutant_scores.append(mut_score)
            ref_scores.append(ref_score)

        batch_scores = (self.esm_score_weights['mutant'] * np.array(mutant_scores) +
                        self.esm_score_weights['reference'] * np.array(ref_scores))
        return batch_scores

    def _initialize_sampler(self, score_matrix):
        """
        Initialize the sampler for generating mutant sequences.

        Args:
            score_matrix (np.ndarray): Initial score matrix.

        Returns:
            tuple: Sampler, sum of scores, sum of squared scores, and observation counts.
        """
        if score_matrix is None:
            print(f"Computing initial score matrix, using batches of {self.init_score_batch_size}.")
            score_matrix = self._compute_single_mutant_score_matrix(n_batch=self.init_score_batch_size)
        self.initial_score_matrix = score_matrix.copy()
        self.score_matrix = score_matrix.copy()
        if self.normalize_scores:
            self.ref_score_value = self.optimizer_params.get('ref_score_value', None)
            self.ref_score_scale = self.optimizer_params.get('ref_score_scale', None)
            if self.ref_score_value is None:
                #self.ref_score_value = score_matrix.flatten().mean()
                self.ref_score_value = np.quantile(score_matrix.flatten(), 0.8)
            if self.ref_score_scale is None:
                self.ref_score_scale = score_matrix.flatten().std()
            print(f"Reference score value: {self.ref_score_value:.4f}, std dev: {self.ref_score_scale:.4f}. To normalize scores.")
            self.score_matrix = (score_matrix.copy() - self.ref_score_value) / (self.ref_score_scale + 1.0)
        self.sum_of_scores_matrix = score_matrix.copy()
        self.mut_to_num_seqs_matrix = np.ones_like(self.sum_of_scores_matrix)
        self.sum_of_scores_squared_matrix = np.square(self.sum_of_scores_matrix)

        sampler = CombinatorialMutationSampler(
            sequence=self.ref_seq,
            forbidden_sites=self.sites_to_ignore,
            temperature=self.T,
            sampling="boltzmann",
            score_matrix=self.score_matrix,
            verbose=True
        )
        return sampler

    def _compute_single_mutant_score_matrix(self, n_batch=500):
        """
        Compute the initial score matrix by scoring all single mutants in batches. Relevant when score_matrix is None.

        Args:
            n_batch (int, optional): Batch size for processing single mutants. Default is 500.

        Returns:
            np.ndarray: Score matrix for single mutants.
        """
        print(f"Scoring all single mutants to compute initial score matrix.")
        single_mutant_sequences = generate_single_mutants(ref_seq=self.ref_seq, sites_to_exclude = self.sites_to_ignore)
        L = len(self.ref_seq) - len(self.sites_to_ignore)
        num_single_mutants = L * 19
        assert num_single_mutants == len(single_mutant_sequences), "The number of single mutants is wrong."
        print(f"Computing scores for {len(single_mutant_sequences)} single mutants.")
        all_scores = []
        for i in range(0, len(single_mutant_sequences), n_batch):
            batch_sequences = single_mutant_sequences[i:i + n_batch]
            batch_tokens, rel_seqs_tensors_padded = self.prepare_batch(batch_sequences, self.device)
            if self.esm_only:
                batch_scores = self._compute_esm_scores(batch_tokens, rel_seqs_tensors_padded)
            else:
                with torch.no_grad():
                    new_predictions = self.model(batch_tokens, rel_seqs_tensors_padded)
                batch_scores = compute_model_scores(new_predictions, self.task_weights).squeeze().detach().cpu().numpy()
                del new_predictions
            all_scores.extend(batch_scores)
            print(f"Finished sequence {i} of {len(single_mutant_sequences)}.")
            del batch_sequences, batch_tokens, rel_seqs_tensors_padded
        # Store mean and variance of single mutants scores as reference
        if len(all_scores) == 0:
            raise ValueError("No scores were computed: error.")
        x = np.array(all_scores)
        return create_score_matrix(self.ref_seq, single_mutant_sequences, all_scores, wt_scores=0.0)

    def _set_seeds(self):
        """
        Set random seeds for reproducibility.
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.num_gpus > 1:
            torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups

    def _update_observation_matrices(self, sequences, scores):
        """
        Update the sum_of_scores_matrix and mut_to_num_seqs_matrix based on the new sequences and their scores.

        Args:
            sequences (list): List of sequences that were scored.
            scores (list): Corresponding scores for the sequences.
        """
        for seq, score in zip(sequences, scores):
            mutations = seq.split('-')
            for mutation in mutations:
                wt_aa = mutation[0]
                site = int(mutation[1:-1]) - 1  # Convert to 0-based index
                mut_aa = mutation[-1]
                mut_key = (AA_TO_IDX[mut_aa], site)

                # Update the sum_of_scores_matrix and mut_to_num_seqs_matrix
                self.sum_of_scores_matrix[mut_key] += score
                self.mut_to_num_seqs_matrix[mut_key] += 1
                self.sum_of_scores_squared_matrix[mut_key] += score * score

    def detect_phase_transition(self, i):
        """
        Detect a phase transition based on the current mean and variance of scores.

        Args:
            i (int): Current iteration.
        """
        cur_mean_avg = np.mean(np.array(self.all_mean_ml_scores[-5:]))
        self.last_high_var = max(np.convolve(self.all_variances, np.ones(5)/5, mode='valid'))
        if (cur_mean_avg > self.score_threshold):
            self.last_phase_transition = i
            self.last_high_ml_score = max(np.convolve(self.all_mean_ml_scores, np.ones(5)/5, mode='valid'))
            print(f"#### Phase transition detected. High ml score avg: {self.last_high_ml_score:.3g}, high var: {self.last_high_var:.3g}. ####")
            self.active_phase_transition = True
            self.num_phase_transitions += 1
            if self.first_phase_transition is None:
                self.first_phase_transition = i
            print(f"Will start increasting temperature in {self.patience_phase_trans} iterations, unless a phase reversal is detected first.")

    def detect_phase_transition_reversal(self, i):
        """
        Detect a phase transition reversal based on the current mean and variance of scores.

        Args:
            i (int): Current iteration.
        """
        cur_var_avg = np.mean(np.array(self.all_variances[-2:]))
        cur_mean_avg = np.mean(np.array(self.all_mean_ml_scores[-2:]))
        if cur_mean_avg < self.reversal_threshold:
        #if (cur_mean_avg < (self.last_high_ml_score + self.initial_ml_score) / 2) and cur_var_avg > (self.last_high_var * 0.5):
            print(f"####### Phase transition reversal detected. ########")
            self.active_phase_transition = False
        if (i > self.last_phase_transition + self.patience):
            print(f"####### Stopping phase transition because we have not detected a phase reversal in 15 iterations. ########")
            self.active_phase_transition = False

    def update_temperature(self, i, num_sequences):
        """
        Update the temperature based on the current number of sequences and phase transition state, along with cooling and heating cycles.

        Args:
            i (int): Current iteration.
            num_sequences (int): Number of sequences sampled in the current iteration.
        """
        if self.T < 0.01:
            self.T *= 1.4
            print(f"Increasing temperature because it got too low, to prevent sampler errors.")
        elif num_sequences < np.round(self.n_batch * 0.5):  # if too few sequences, increase temperature
            if self.T < 10 * self.optimizer_params['T']:
                self.low_temp_threshold = self.T * 1.10
                self.T *= 1.4
                print(f"#### Increased temperature to {self.T:.3g} to find more sequences per iteration. ####")
                print(f"#### Low T threshold set to {self.low_temp_threshold:3g}. ####")
        elif self.simple_simulated_annealing:
            self.T *= self.cooling_rate
        elif self.cool_then_heat:
            # Initialize counters and phase states if not already done
            if not hasattr(self, 'heating_phase'):
                self.heating_phase = False  # Start with cooling
                self.cooling_iterations = 0
                self.heating_iterations = 0
            if self.heating_phase:
                # Heating phase: heat until temperature reaches 2.5
                if (self.heating_iterations >= 45) or (self.T >= self.high_temp_threshold):
                    print("### Staring to cool the system. ###")
                    self.heating_phase = False  # Switch to cooling
                    self.cooling_iterations = 1  # Reset cooling iteration counter
                    self.T *= self.cooling_rate
                else:
                    self.T /= self.cooling_rate  # Continue heating
                    self.heating_iterations += 1
            else:
                # Cooling phase: cool for a fixed number of iterations
                if (self.cooling_iterations >= 45) or (self.T <= self.low_temp_threshold):
                    print(f"### Staring to heat the system. T threshold: {self.low_temp_threshold:3g}. T: {self.T:3g}. ###")
                    self.heating_phase = True  # Switch to heating after 120 iterations
                    self.heating_iterations = 1
                    self.T /= self.cooling_rate
                else:
                    self.T *= self.cooling_rate  # Continue cooling
                    self.cooling_iterations += 1  # Increment cooling iteration counter
        elif self.active_phase_transition and (i > self.last_phase_transition + self.patience_phase_trans):
            self.T *= self.heating_rate
            print(f"#### Increasing temperature to {self.T:.3g} to achieve a phase reversal. ####")
        elif self.last_phase_transition is None:
            self.T *= self.cooling_rate # no phase transition detected yet.
        else:
            self.T *= np.square(self.cooling_rate)*self.cooling_rate if self.num_mutations < 4 else np.square(self.cooling_rate)

    def score_sequences_and_update_score_matrices(self, sequences):
        """
        Score the provided sequences, and store them in scored sequences dictionary. Updates matrices to compute
        average mutation scores for sampler too.

        Args:
            sequences (list): List of sequences to be scored, in relative sequence string format.

        Returns:
            tuple: A tuple containing:
                - final_sequences (list): List of sequences with their scores updated.
                - scores (list): List of scores corresponding to the provided sequences.
                - n_new (int): Number of new sequences scored in this iteration.
        """
        new_indices, old_indices = [], []
        for i, seq in enumerate(sequences):
            if seq not in self.scored_sequences:
                new_indices.append(i)
            else:
                old_indices.append(i)
        n_new = len(new_indices)

        # Ensure the number of new sequences is a multiple of the number of GPUs
        excess = n_new % self.num_gpus
        if excess > 0:
            print(f"Dropping {excess} sequences to make the batch size a multiple of the number of GPUs.")
            new_indices = new_indices[:-excess]
            n_new = len(new_indices)
        new_sequences = [sequences[i] for i in new_indices]
        final_sequences = [sequences[i] for i in new_indices + old_indices]
        print(f">> {n_new} out of {len(sequences)} sequences this iteration are new and will be scored.")

        #new_prob_matrix = None
        if n_new > 0:
            batch_tokens, rel_seqs_tensors_padded = self.prepare_batch(new_sequences, self.device)
            if self.esm_only:
                new_scores = self._compute_esm_scores(batch_tokens, rel_seqs_tensors_padded)
            else:
                with torch.no_grad():
                    new_predictions = self.model(batch_tokens, rel_seqs_tensors_padded)
                new_scores = compute_model_scores(new_predictions, self.task_weights).squeeze().detach().cpu().numpy()
            self.update_scored_sequences(new_sequences, new_scores)
            #self._update_mutation_score_map(new_sequences, new_scores)  # only if we want to experiment with using seq scores differently
            self._update_observation_matrices(new_sequences, new_scores)
        scores = [self.scored_sequences[seq] for seq in final_sequences]
        return final_sequences, scores, n_new

    def _update_mutation_score_map(self, sequences, scores):
        """
        Update the mutation score map with the newly scored sequences.

        Args:
            sequences (list): List of sequences that were scored.
            scores (list): Corresponding scores for the sequences.
        """
        for seq, score in zip(sequences, scores):
            mutations = seq.split('-')
            for mutation in mutations:
                wt_aa = mutation[0]
                site = int(mutation[1:-1]) - 1  # Convert to 0-based index
                mut_aa = mutation[-1]
                mut_key = (AA_TO_IDX[mut_aa], site)

                if mut_key not in self.mutation_score_map:
                    self.mutation_score_map[mut_key] = []
                self.mutation_score_map[mut_key].append(score)

    def update_scored_sequences(self, sequences, scores):
        """
        Update the scored sequences dictionary with new sequences and their scores. Assumes only newly visited sequences are passed.

        Args:
            sequences (list): List of sequences.
            scores (list): List of scores corresponding to the sequences.
        """
        for seq, score in zip(sequences, scores):
            self.scored_sequences[seq] = score

    def profile_optimize(self):
        """
        Optimize the protein sequences with profiling enabled for performance analysis.

        Returns:
            tuple: DataFrame of results and DataFrame of statistics.
        """
        # Just for profiling purposes. Then delete
        pr = cProfile.Profile()
        pr.enable()

        df, df_stats = self.optimize()  # Call the method you want to profile

        pr.disable()
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        return df, df_stats  # Return the results as usual

    def optimize(self):
        """
        Optimize the protein sequences using simulated annealing, and logic to detect phase transitions and reversals. This is the main
        class method, where all the action happens.

        Returns:
            tuple: DataFrame of results and DataFrame of statistics.
        """
        for i in range(self.n_seqs // self.n_batch):
            # Compute N_eff using the current prob_matrix
            Neff = compute_Neff_for_probmatrix(self.sampler.prob_matrix)
            # Sample sequences to process this iteration
            if self.sample_variety_of_mutation_numbers: # Sample sequences with a mix of number of mutations
                sequences = self.sampler.sample_mutant_library(library_size=self.n_batch,
                                                               mutation_proportions=self.num_sequences_proportions,
                                                               discard_bad_sequences=False, dedupe=True)
            else:   # To sample one number of mutations at a time only: default
                sequences = self.sampler.sample_multi_mutants(num_mutations=self.num_mutations, library_size=self.n_batch,
                                                              discard_bad_sequences=True, dedupe=True)
            sequences = list(set(sequences))
            if len(sequences) < 1:
                print(f"Sampler failed to generate any new sequences in iteration {i}. Increasing temperature and trying again.")
                self.update_temperature(i, len(sequences))
                self.sampler.update_boltzmann_distribution(new_temperature=self.T, new_score_matrix=self.score_matrix)
                continue
            self.iters_for_seqs.extend(np.repeat(i, len(sequences)))
            print(f"Starting iter {i}: processing {len(sequences)} sequences, using T = {self.T:.2g}.")
            n1, n2, n3 = Neff['Neff'], Neff['Neff_cols'], Neff['Neff_rows']
            print(f"Joint probability has {n1:.3g} effective entries: {n2:.3g} sites, {n3:.3g} amino acids.")

            # Score sequences; update sampler already using only newly seen sequences
            sequences, scores, n_new = self.score_sequences_and_update_score_matrices(sequences)

            # Store sequences and scores
            self.all_sequences.extend(sequences)
            self.all_scores.extend(scores)
            self.all_variances.append(np.var(scores))
            self.all_mean_ml_scores.append(np.mean(scores))
            self.n_eff_joint.append(n1)
            self.n_eff_sites.append(n2)
            self.n_eff_aa.append(n3)
            self.all_Ts.append(self.T)
            self.all_num_sampled_seqs.append(len(sequences))
            self.all_num_new_seqs.append(n_new)
            self.all_phase_transition_numbers.append(self.num_phase_transitions)

            if i == 10:
                self.initial_var = np.mean(np.array(self.all_variances))
                self.initial_ml_score = np.mean(np.array(self.all_mean_ml_scores))
                print(f"Baseline ml score variance is {self.initial_var:.3g}, baseline avg is {self.initial_ml_score:.3g}.")

            # Detect phase transition
            if ((self.active_phase_transition is False) and (i > 10) and (self.cool_then_heat is False) and
                (self.simple_simulated_annealing is False)):
                self.detect_phase_transition(i)

            # Detect phase transition reversal
            if ((self.active_phase_transition) and (i > self.last_phase_transition)
                and (self.simple_simulated_annealing is False) and (self.cool_then_heat is False)):
                self.detect_phase_transition_reversal(i)

            # Update temperature
            self.update_temperature(i, len(sequences))

            # Need to use the new score matrix only after new temperature has been set
            if n_new > 0: # only update when new sequences are observed.
                # Update the sampler's score matrix
                self.score_matrix = self.sum_of_scores_matrix / (self.mut_to_num_seqs_matrix + EPS)
                if self.boost_mutations_with_high_variance and (self.gamma > 0.0):
                    var_matrix = self.sum_of_scores_squared_matrix / (self.mut_to_num_seqs_matrix + EPS) - np.square(self.score_matrix)
                    var_matrix = np.clip(var_matrix, a_min=0, a_max=None) # remove any negative values
                    self.score_matrix += self.gamma*np.sqrt(var_matrix)
                if self.normalize_scores:
                    self.score_matrix = (self.score_matrix - self.ref_score_value) / (self.ref_score_scale + 1.0)

            self.sampler.update_boltzmann_distribution(new_temperature=self.T, new_score_matrix=self.score_matrix)

            str1 = f" of {self.all_mean_ml_scores[-1]:.3g}, and std. dev. {np.sqrt(self.all_variances[-1]):.3g}."
            print(f"We have explored {len(self.all_sequences)} sequences. Finished iter {i} with mean ml score" + str1)

        print(f"Visited {len(self.all_sequences)} sequences out of {self.n_seqs} intended.")
        self._deduplicate_results()
        print(f"There were {len(self.all_sequences)} unique sequences visited.")

        df, df_stats = self.prepare_results(n_seqs_to_keep=self.n_seqs_to_keep)
        self.df, self.df_stats = df, df_stats
        return copy.deepcopy(df), copy.deepcopy(df_stats)

    def _deduplicate_results(self):
        """
        Deduplicate the results to keep unique sequences and their mean scores.
        """
        unique_sequences = {}
        sequence_counts = {}
        first_seen_iteration = {}
        for sequence, score, iteration in zip(self.all_sequences, self.all_scores, self.iters_for_seqs):
            if sequence not in unique_sequences:
                unique_sequences[sequence] = score
                sequence_counts[sequence] = 1
                first_seen_iteration[sequence] = iteration
            else:
                unique_sequences[sequence] += score  # we'll compute the mean
                sequence_counts[sequence] += 1 # number of iterations where a sequence appears
        # Second pass: Calculate the mean score for each sequence
        for sequence in unique_sequences:
            unique_sequences[sequence] = unique_sequences[sequence] / sequence_counts[sequence]
        self.all_sequences = list(unique_sequences.keys())
        self.all_scores = list(unique_sequences.values())
        self.seq_iter_counts = list(sequence_counts.values())
        self.iter_first_seen = list(first_seen_iteration.values())

    def prepare_results(self, n_seqs_to_keep=None):
        """
        Prepare the results by sorting and selecting the top sequences.

        Args:
            n_seqs_to_keep (int, optional): Number of sequences to keep. Default is None (keep all).

        Returns:
            tuple: DataFrame of results and DataFrame of statistics.
        """
        if n_seqs_to_keep is None:
            n_seqs_to_keep = len(self.all_scores)  # keep all if None
        sorted_indices = np.argsort(self.all_scores)[::-1][:n_seqs_to_keep]
        ranked_sequences = [self.all_sequences[i] for i in sorted_indices]
        ranked_scores = [self.all_scores[i] for i in sorted_indices]
        ranked_counts = [self.seq_iter_counts[i] for i in sorted_indices]
        ranked_first_seen = [self.iter_first_seen[i] for i in sorted_indices]

        df = pd.DataFrame({
            'sequences': np.array(ranked_sequences),
            'ml_score': np.array(ranked_scores),
            'counts': np.array(ranked_counts, dtype=int),
            'num_mutations': np.array(np.ones(len(ranked_sequences), dtype=int) * self.num_mutations, dtype=int),
            'iteration': np.array(ranked_first_seen, dtype=int)
        })

        df_stats = pd.DataFrame({
            'iteration': np.array(range(len(self.all_mean_ml_scores)), dtype=int),
            'avg_ml_score': np.array(self.all_mean_ml_scores),
            'var_ml_score': np.array(self.all_variances),
            'n_eff_joint': np.array(self.n_eff_joint),
            'n_eff_sites': np.array(self.n_eff_sites),
            'n_eff_aa': np.array(self.n_eff_aa),
            'T': np.array(self.all_Ts),
            'n_seqs': np.array(self.all_num_sampled_seqs, dtype=int),
            'n_new_seqs': np.array(self.all_num_new_seqs, dtype=int),
            'num_phase_transitions': np.array(self.all_phase_transition_numbers, dtype=int)
        })
        return df, df_stats

    def save_results(self, filename=None, n_seqs_to_keep=10000):
        """
        Save the results to a CSV file.

        Args:
            filename (str, optional): Filename for saving the results. Default is None.
            n_seqs_to_keep (int, optional): Number of sequences to keep in the results. Default is 10000.
        """
        if n_seqs_to_keep is not None:
            df, df_stats = self.prepare_results(n_seqs_to_keep)
        else:
            df, df_stats = self.df, self.df_stats # computed before for all sequences
        if filename is None:
            filename = f'optimizer_{self.num_mutations}_mutations.csv'
        filename = 'ranked_sequences_' + filename
        df.to_csv(filename, index=False, float_format='%.5g')
        stats_filename = filename.replace('ranked_sequences', 'optimizer_stats')
        df_stats.to_csv(stats_filename, index=False, float_format='%.5g')

    def plot_scores(self, save_figs=True):
        """
        Plot the scores and statistics of the optimization process.

        Args:
            save_figs (bool, optional): Whether to save the figures. Default is True.
        """
        df, df_stats = self.df, self.df_stats
        df_stats['std_dev_ml_score'] = np.sqrt(df_stats['var_ml_score'])

        sns.set(style="whitegrid")

        # First Group: Statistics by iteration
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

        # Plot avg_ml_score and var_ml_score
        ax1 = sns.lineplot(x='iteration', y='avg_ml_score', data=df_stats, ax=axes[0], color = 'b', label='avg ml score', linewidth=2.5)
        ax2 = ax1.twinx()
        sns.lineplot(x='iteration', y='std_dev_ml_score', data=df_stats, ax=ax2, color='r', label='ml score std dev', linewidth=2.5)
        ax1.set_ylabel('Average ML Score', color=ax1.lines[0].get_color())
        ax2.set_ylabel('ML Score Std. Dev.', color='r')
        ax1.tick_params(axis='y', labelcolor=ax1.lines[0].get_color())
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Plot n_eff_sites and n_eff_aa
        ax3 = sns.lineplot(x='iteration', y='n_eff_sites', data=df_stats, ax=axes[1], color = 'b', label=f'site $n_{{eff}}$', linewidth=2.5)
        ax4 = ax3.twinx()
        sns.lineplot(x='iteration', y='n_eff_aa', data=df_stats, ax=ax4, color='r', label=f'aa $n_{{eff}}$', linewidth=2.5)
        ax3.set_ylabel(f'site $n_{{eff}}$', color=ax3.lines[0].get_color())
        ax4.set_ylabel(f'amino acid $n_{{eff}}$', color='r')
        ax3.tick_params(axis='y', labelcolor=ax3.lines[0].get_color())
        ax4.tick_params(axis='y', labelcolor='r')
        ax3.legend(loc='upper left')
        ax4.legend(loc='upper right')

        # Plot temperatures
        ax5 = sns.lineplot(x='iteration', y='T', data=df_stats, ax=axes[2], linewidth=2.5)
        ax5.set_ylabel('Temperature')
        ax5.set_xlabel('Iteration')

        plt.title(f'Num Mutations {self.optimizer_params["num_mutations"]}.')

        # Identify iterations where phase transitions occur
        phase_transition_iters = df_stats['iteration'][df_stats['num_phase_transitions'].diff() == 1].values

        # Add vertical dashed lines to the first group of plots
        for ax in [ax1, ax3, ax5]:
            for pt_iter in phase_transition_iters:
                ax.axvline(x=pt_iter, color='black', linestyle='--', linewidth=1)

        plt.tight_layout()
        if save_figs:
            plt.savefig(f'iteration_statistics_num_mutations_{self.optimizer_params["num_mutations"]}.png', bbox_inches='tight')
        plt.show()

        # Second Group: Scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Define marker styles based on the number of phase transitions, using '.' for any additional values
        marker_styles = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'P', 5: '*', 6: 'X', 7: 'v', 8: '<', 9: '>'}
        default_marker = '.'
        # Create a dictionary mapping number of phase transitions to marker style
        num_phase_transition_markers = {num: marker_styles.get(num, default_marker) for num in df_stats['num_phase_transitions'].unique()}

        # Temperature vs avg_ml_score and var_ml_score
        ax1 = axes[0]
        ax2 = ax1.twinx()
        ax2.grid(False)  # Disable the grid for the secondary y-axis

        for num_phase_transitions, group in df_stats.groupby('num_phase_transitions'):
            marker = num_phase_transition_markers[num_phase_transitions]
            sns.scatterplot(x='T', y='avg_ml_score', data=group, ax=ax1, label = '', color='b', s=50, marker=marker, alpha=0.85)
            sns.scatterplot(x='T', y='std_dev_ml_score', data=group, ax=ax2, label = '', color='r', s=50, marker=marker, alpha=0.85)
        ax1.set_ylabel('Average ML Score', color='b')
        ax2.set_ylabel('ML Std. Dev.', color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.set_xlabel('Temperature')
        #ax1.legend(loc='upper left')
        #ax2.legend(loc='upper right')

        # Temperature vs n_eff_sites and n_eff_aa
        ax3 = axes[1]
        ax4 = ax3.twinx()
        ax4.grid(False)  # Disable the grid for the secondary y-axis

        for num_phase_transitions, group in df_stats.groupby('num_phase_transitions'):
            marker = num_phase_transition_markers[num_phase_transitions]
            sns.scatterplot(x='T', y='n_eff_sites', data=group, ax=ax3, label = '', color='b', s=50, marker=marker, alpha=0.85)
            sns.scatterplot(x='T', y='n_eff_aa', data=group, ax=ax4, label = '', color='r', s=50, marker=marker, alpha=0.85)
        ax3.set_ylabel(f'Site $n_{{eff}}$', color='b')
        ax4.set_ylabel(f'AA $n_{{eff}}$', color='r')
        ax3.tick_params(axis='y', labelcolor='b')
        ax4.tick_params(axis='y', labelcolor='r')
        ax3.set_xlabel('Temperature')
        #ax3.legend(loc='upper left')
        #ax4.legend(loc='upper right')

        plt.title(f'Num Mutations {self.optimizer_params["num_mutations"]}.')
        plt.tight_layout()
        if save_figs:
            plt.savefig(f'temperature_scatter_plots_num_mutations_{self.optimizer_params["num_mutations"]}.png', bbox_inches='tight')
        plt.show()

        # Third Group: Density plots

        # Density plots for early and late iterations
        if self.first_phase_transition is not None:
            median_iteration = self.first_phase_transition
        else:
            median_iteration = df['iteration'].median()

        print(median_iteration)
        print(self.first_phase_transition)
        early_df = df[df['iteration'] <= median_iteration]
        late_df = df[df['iteration'] > median_iteration]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        #sns.histplot(df['ml_score'], bins=30, cumulative=False, stat='density',
        #             kde=False, element='step', color='orange', label='All ML Scores', ax=ax)
        sns.histplot(early_df['ml_score'], bins=30, cumulative=False, stat='density',
                     kde=False, element='step', color='blue', label='Early Iterations', ax=ax)
        sns.histplot(late_df['ml_score'], bins=30, cumulative=False, stat='density', kde=False,
                     element='step', color='orange', label='Late Iterations', ax=ax)

        # Calculate the output of the wildtype sequence
        plt.axvline(x=self.wt_score, color='black', linestyle='--', linewidth=2.5, label='WT Score')

        plt.title(f'Num Mutations {self.optimizer_params["num_mutations"]}.')
        plt.xlabel('ML Model Score')
        plt.ylabel('Density')
        plt.legend()
        if save_figs:
            plt.savefig(f'score_density_plots_num_mutations_{self.optimizer_params["num_mutations"]}.png', bbox_inches='tight')
        plt.show()
