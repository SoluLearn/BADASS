import numpy as np
import random
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Callable
from badass.mld.sequence_sampler import CombinatorialMutationSampler
from badass.utils.sequence_utils import (
    AMINO_ACIDS,
    AA_TO_IDX,
    EPS,
    rel_sequences_to_dict,
    apply_mutations,
    generate_single_mutants,
    compute_Neff_for_probmatrix,
    create_score_matrix
)

import logging
logger = logging.getLogger(__name__)

class GeneralProteinOptimizer:
    """
    Model-agnostic protein sequence optimizer using simulated annealing with phase transition detection.
    
    Args:
        predictor (Callable): Function that takes a protein sequence string and returns a scalar score
        ref_seq (str): Reference/wild-type protein sequence
        optimizer_params (dict): Dictionary containing parameters for the optimization process:
            - seqs_per_iter (int): Number of sequences to evaluate per iteration
            - num_iter (int): Number of iterations to run
            - init_score_batch_size (int): Batch size for initial scoring of single mutants
            - T (float): Initial temperature for simulated annealing
            - seed (int): Random seed
            - gamma (float): Weight for variance boosting
            - cooling_rate (float): Rate at which temperature decreases
            - num_mutations (int): Number of mutations per sequence
            - sites_to_ignore (List[int]): Sites to exclude from mutation
            - normalize_scores (bool): Whether to normalize scores
            - simple_simulated_annealing (bool): Use simple SA without adaptation
            - cool_then_heat (bool): Use cooling-then-heating schedule
        score_matrix (np.ndarray, optional): Initial score matrix. Default is None.
    
    Example:
        def predict_protein(sequence: str) -> float:
            # Custom prediction function
            return score
            
        optimizer = GeneralProteinOptimizer(
            predictor=predict_protein,
            ref_seq="MKKLVIV...",
            optimizer_params={
                'seqs_per_iter': 500,
                'num_iter': 100,
                'T': 2.0,
                'seed': 42,
                'num_mutations': 5,
                'sites_to_ignore': [1, 330]
            }
        )
        df, df_stats = optimizer.optimize()
        optimizer.plot_scores()
        optimizer.save_results()
    """
    def __init__(
        self,
        predictor: Callable[[str], float],
        ref_seq: str,
        optimizer_params: dict,
        score_matrix: np.ndarray = None,
    ):
        self.predictor = predictor
        self.ref_seq = ref_seq
        self.optimizer_params = optimizer_params
        self._setup_optimizer_params()
        self._set_seeds()
        
        # Initialize sampler with score matrix
        self.wt_score = self.predictor([self.ref_seq])[0]
        self.sampler = self._initialize_sampler(score_matrix)

        # Get reference sequence score
        self._set_initial_score_thresholds()
        

    def _setup_optimizer_params(self):
        """Setup optimization parameters and initialize tracking variables"""
        # Core parameters
        self.normalize_scores = self.optimizer_params.get('normalize_scores', True)
        self.T = self.optimizer_params.get('T', 1.5) 
        self.n_batch = self.optimizer_params.get('seqs_per_iter', 500)
        self.init_score_batch_size = self.optimizer_params.get('init_score_batch_size', 500)
        self.total_iter = self.optimizer_params.get('num_iter', 200)
        self.n_seqs = self.total_iter * self.n_batch
        self.seed = self.optimizer_params.get('seed', 7)
        self.gamma = self.optimizer_params.get('gamma', 0.5)
        self.boost_mutations_with_high_variance = self.optimizer_params.get('boost_mutations_with_high_variance', True)
        self.cooling_rate = self.optimizer_params.get('cooling_rate', 0.92)
        self.num_mutations = self.optimizer_params.get('num_mutations', 5)
        self.sites_to_ignore = self.optimizer_params.get('sites_to_ignore', [1])
        self.simple_simulated_annealing = self.optimizer_params.get('simple_simulated_annealing', False)
        self.cool_then_heat = self.optimizer_params.get('cool_then_heat', False)
        self.n_seqs_to_keep = self.optimizer_params.get('n_seqs_to_keep', None)
        self.adaptive_upper_threshold = self.optimizer_params.get('adaptive_upper_threshold', None)
        
        # Initialize tracking variables
        self.all_sequences = []
        self.all_scores = []
        self.iters_for_seqs = []
        self.n_eff_joint = []
        self.n_eff_sites = []
        self.n_eff_aa = []
        self.all_Ts = []
        self.seq_iter_counts = []
        self.all_variances = []
        self.all_num_new_seqs = []
        self.all_num_sampled_seqs = []
        self.all_mean_ml_scores = []
        self.all_phase_transition_numbers = []
        self.scored_sequences = {}
        
        # Phase transition tracking
        self.active_phase_transition = False
        self.first_phase_transition = None
        self.initial_var = None
        self.initial_ml_score = None
        self.last_high_ml_score = float('-inf')
        self.last_high_var = float('-inf')
        self.last_phase_transition = None
        self.num_phase_transitions = 0
        self.patience_phase_trans = 3
        self.patience = 10 if (self.num_mutations > 3) else 15
        self.patience += self.patience_phase_trans
        self.heating_rate = 1.4 if (self.num_mutations > 3) else 1.6
        self.low_temp_threshold = 0.02
        self.high_temp_threshold = 1.6
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def _set_initial_score_thresholds(self):
        """
        Set thresholds for phase transition detection using single mutant data.
        If thresholds are provided in optimizer_params, uses those instead.
        """
        self.score_threshold = self.optimizer_params.get('score_threshold', None)
        if self.score_threshold is None:
            # Use single mutant score distribution to set threshold
            single_mutant_scores = self.initial_score_matrix.flatten()
            score_mean = np.mean(single_mutant_scores)
            score_std = np.std(single_mutant_scores)
            self.score_threshold = score_mean + score_std
            logger.info(f"Attempting to compute reasonable phase transition score threshold to {self.score_threshold:.3g}.")
            
        self.reversal_threshold = self.optimizer_params.get('reversal_threshold', None)
        if self.reversal_threshold is None:
            # Set reversal threshold below mean of single mutant scores
            self.reversal_threshold = np.mean(self.initial_score_matrix.flatten())
            logger.info(f"Setting phase reversal threshold to {self.reversal_threshold:.3g}.")

    def _get_adaptive_score_threshold(self):
        """
        Compute adaptive score threshold based on current scores.
        
        If float, return the quantile score. If int, return the score of the top -Nth sequence.
        """
        if isinstance(self.adaptive_upper_threshold, float):
            return np.quantile(self.all_scores, self.adaptive_upper_threshold)
        elif isinstance(self.adaptive_upper_threshold, int):
            return np.sort(self.all_scores)[::-1][self.adaptive_upper_threshold]
        else:
            raise ValueError("Invalid adaptive_upper_threshold type.")

    # Required method stubs - to be implemented
    def _initialize_sampler(self, score_matrix):
        """
        Initialize the sampler for generating mutant sequences.

        Args:
            score_matrix (np.ndarray, optional): Initial score matrix for mutation sampling.
                If None, computes it using single mutant scoring.

        Returns:
            CombinatorialMutationSampler: Initialized sampler object
        """
        # Compute score matrix if not provided
        if score_matrix is None:
            logger.info(f"Computing initial score matrix, using batches of {self.init_score_batch_size}.")
            score_matrix = self._compute_single_mutant_score_matrix(n_batch=self.init_score_batch_size)
        
        # Store initial and working copies of score matrix
        self.initial_score_matrix = score_matrix.copy()
        self.score_matrix = score_matrix.copy()
        
        # Initialize score normalization if enabled
        if self.normalize_scores:
            self.ref_score_value = self.optimizer_params.get('ref_score_value', None)
            self.ref_score_scale = self.optimizer_params.get('ref_score_scale', None)
            
            if self.ref_score_value is None:
                self.ref_score_value = np.quantile(score_matrix.flatten(), 0.8)
            if self.ref_score_scale is None:
                self.ref_score_scale = score_matrix.flatten().std()
                
            logger.info(f"Reference score value: {self.ref_score_value:.4f}, std dev: {self.ref_score_scale:.4f}.")
            self.score_matrix = (score_matrix.copy() - self.ref_score_value) / (self.ref_score_scale + 1.0)
        
        # Initialize matrices for tracking scores and observations
        self.sum_of_scores_matrix = score_matrix.copy()
        self.mut_to_num_seqs_matrix = np.ones_like(self.sum_of_scores_matrix)
        self.sum_of_scores_squared_matrix = np.square(self.sum_of_scores_matrix)
        
        # Create and return sampler
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
        Compute the initial score matrix by scoring all single mutants in batches.
        
        Args:
            n_batch (int): Batch size for processing single mutants

        Returns:
            np.ndarray: Score matrix of shape (20, L) where L is sequence length
        """
        # Generate all possible single mutant sequences
        single_mutant_sequences = generate_single_mutants(ref_seq=self.ref_seq, 
                                                        sites_to_exclude=self.sites_to_ignore)
        
        # Verify expected number of single mutants
        L = len(self.ref_seq) - len(self.sites_to_ignore)
        num_single_mutants = L * 19  # 19 possible mutations at each position
        assert num_single_mutants == len(single_mutant_sequences), "Incorrect number of single mutants"
        logger.info(f"Computing scores for {len(single_mutant_sequences)} single mutants.")
        
        # Score mutants in batches
        all_scores = []
        for i in range(0, len(single_mutant_sequences), n_batch):
            batch_sequences = single_mutant_sequences[i:i + n_batch]
            # Convert relative mutation strings to full sequences
            abs_seqs = [apply_mutations(self.ref_seq, 
                                      rel_sequences_to_dict([seq], sep='-')[0]) 
                       for seq in batch_sequences]
            # Score the sequences using the predictor
            batch_scores = self.predictor([seq for seq in abs_seqs])
            all_scores.extend(batch_scores)
            logger.info(f"Finished sequence {i} of {len(single_mutant_sequences)}.")
            
        if len(all_scores) == 0:
            raise ValueError("No scores were computed")
            
        # Create and return score matrix
        return create_score_matrix(self.ref_seq, single_mutant_sequences, 
                                 all_scores, wt_scores=self.wt_score)
    
    def update_scored_sequences(self, sequences, scores):
        """
        Update the cache of scored sequences.
        
        Args:
            sequences (list): List of sequences in relative mutation notation
            scores (list): Corresponding scores for each sequence
        """
        for seq, score in zip(sequences, scores):
            self.scored_sequences[seq] = score
            
    def _update_observation_matrices(self, sequences, scores):
        """
        Update the matrices used for computing mutation probabilities.
        
        Updates three matrices:
        - sum_of_scores_matrix: Sum of scores for each mutation
        - mut_to_num_seqs_matrix: Count of sequences with each mutation
        - sum_of_scores_squared_matrix: Sum of squared scores for variance computation
        
        Args:
            sequences (list): List of sequences in relative mutation notation
            scores (list): Corresponding scores for each sequence
        """
        for seq, score in zip(sequences, scores):
            mutations = seq.split('-')
            for mutation in mutations:
                # Parse mutation string
                wt_aa = mutation[0]
                site = int(mutation[1:-1]) - 1  # Convert to 0-based index
                mut_aa = mutation[-1]
                mut_key = (AA_TO_IDX[mut_aa], site)
                
                # Update tracking matrices
                self.sum_of_scores_matrix[mut_key] += score
                self.mut_to_num_seqs_matrix[mut_key] += 1
                self.sum_of_scores_squared_matrix[mut_key] += score * score
        
    def score_sequences_and_update_score_matrices(self, sequences):
        """
        Score sequences and update the scoring matrices.
        
        Args:
            sequences (list): List of sequences in relative mutation notation (e.g. "A1M-G4P")
            
        Returns:
            tuple: (final_sequences, scores, n_new)
                - final_sequences (list): List of processed sequences
                - scores (list): Corresponding scores
                - n_new (int): Number of new sequences scored
        """
        # Identify new sequences that need scoring
        new_indices, old_indices = [], []
        for i, seq in enumerate(sequences):
            if seq not in self.scored_sequences:
                new_indices.append(i)
            else:
                old_indices.append(i)
        n_new = len(new_indices)
        
        new_sequences = [sequences[i] for i in new_indices]
        final_sequences = [sequences[i] for i in new_indices + old_indices]
        logger.info(f">> {n_new} out of {len(sequences)} sequences this iteration are new and will be scored.")
        
        if n_new > 0:
            # Convert relative mutations to full sequences for new sequences
            abs_seqs = [apply_mutations(self.ref_seq, 
                                      rel_sequences_to_dict([seq], sep='-')[0]) 
                       for seq in new_sequences]
            
            # Score the sequences
            new_scores = self.predictor([seq for seq in abs_seqs])
            
            # Update the scored sequences dictionary
            self.update_scored_sequences(new_sequences, new_scores)
            
            # Update observation matrices for sampling
            self._update_observation_matrices(new_sequences, new_scores)
            
        # Get scores for all sequences (new and old)
        scores = [self.scored_sequences[seq] for seq in final_sequences]
        
        return final_sequences, scores, n_new
        
    def optimize(self):
        """
        Main optimization loop using simulated annealing with phase transition detection.
        
        The method performs iterative sampling and scoring of sequences, adapting the
        temperature and sampling distribution based on observed scores and phase transitions.
        
        Returns:
            tuple: (df, df_stats)
                - df (pd.DataFrame): Results dataframe with sequences and scores
                - df_stats (pd.DataFrame): Statistics about the optimization process
        """
        for i in range(self.n_seqs // self.n_batch):
            # Compute effective number of samples using current probability matrix
            Neff = compute_Neff_for_probmatrix(self.sampler.prob_matrix)
            
            # Sample sequences to evaluate this iteration
            sequences = self.sampler.sample_multi_mutants(
                num_mutations=self.num_mutations, 
                library_size=self.n_batch,
                discard_bad_sequences=True, 
                dedupe=True
            )
            sequences = list(set(sequences))
            
            # Check if sampling was successful
            if len(sequences) < 1:
                logger.info(f"Sampler failed to generate sequences in iteration {i}. "
                      f"Increasing temperature and retrying.")
                self.update_temperature(i, len(sequences))
                self.sampler.update_boltzmann_distribution(
                    new_temperature=self.T, 
                    new_score_matrix=self.score_matrix
                )
                continue
                
            # Track iteration for each sequence
            self.iters_for_seqs.extend(np.repeat(i, len(sequences)))
            
            # Log iteration status
            logger.info(f"Starting iter {i}: processing {len(sequences)} sequences, using T = {self.T:.2g}.")
            n1, n2, n3 = Neff['Neff'], Neff['Neff_cols'], Neff['Neff_rows']
            logger.info(f"Joint probability has {n1:.3g} effective entries: {n2:.3g} sites, {n3:.3g} amino acids.")

            # Score sequences and update matrices
            sequences, scores, n_new = self.score_sequences_and_update_score_matrices(sequences)

            # Store iteration results
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

            # Initialize baseline statistics after warm-up period
            if i == 10:
                self.initial_var = np.mean(np.array(self.all_variances))
                self.initial_ml_score = np.mean(np.array(self.all_mean_ml_scores))
                logger.info(f"Baseline score variance is {self.initial_var:.3g}, "
                      f"baseline avg is {self.initial_ml_score:.3g}.")

            # Phase transition detection and management
            if (not self.active_phase_transition and i > 10 and 
                not self.cool_then_heat and not self.simple_simulated_annealing):
                self.detect_phase_transition(i)

            if (self.active_phase_transition and i > self.last_phase_transition and
                not self.simple_simulated_annealing and not self.cool_then_heat):
                self.detect_phase_transition_reversal(i)

            # Update temperature
            self.update_temperature(i, len(sequences))

            # Update sampling distribution if new sequences were observed
            if n_new > 0:
                # Update score matrix based on observations
                self.score_matrix = self.sum_of_scores_matrix / (self.mut_to_num_seqs_matrix + EPS)
                
                # Apply variance boosting if enabled
                if self.boost_mutations_with_high_variance and (self.gamma > 0.0):
                    var_matrix = (self.sum_of_scores_squared_matrix / 
                                (self.mut_to_num_seqs_matrix + EPS) - 
                                np.square(self.score_matrix))
                    var_matrix = np.clip(var_matrix, a_min=0, a_max=None)
                    self.score_matrix += self.gamma * np.sqrt(var_matrix)
                    
                # Normalize scores if enabled
                if self.normalize_scores:
                    self.score_matrix = ((self.score_matrix - self.ref_score_value) / 
                                       (self.ref_score_scale + 1.0))

                # Update sampler's probability distribution
                self.sampler.update_boltzmann_distribution(
                    new_temperature=self.T,
                    new_score_matrix=self.score_matrix
                )

            # Log progress
            str1 = f" of {self.all_mean_ml_scores[-1]:.3g}, and std. dev. {np.sqrt(self.all_variances[-1]):.3g}."
            logger.info(f"We have explored {len(self.all_sequences)} sequences. "
                  f"Finished iter {i} with mean score" + str1)

        # Final processing
        logger.info(f"Visited {len(self.all_sequences)} sequences out of {self.n_seqs} intended.")
        self._deduplicate_results()
        logger.info(f"There were {len(self.all_sequences)} unique sequences visited.")

        # Prepare and store results
        df, df_stats = self.prepare_results(n_seqs_to_keep=self.n_seqs_to_keep)
        self.df, self.df_stats = df, df_stats
        
        return copy.deepcopy(df), copy.deepcopy(df_stats)
        
    def detect_phase_transition(self, i):
        """
        Detect a phase transition based on the current mean and variance of scores.
        
        A phase transition is detected when the mean score exceeds the score threshold,
        indicating the system has found a new high-scoring region of the fitness landscape.
        
        Args:
            i (int): Current iteration number
        """
        # Calculate moving averages
        cur_mean_avg = np.mean(np.array(self.all_mean_ml_scores[-5:]))
        self.last_high_var = max(np.convolve(self.all_variances, np.ones(5)/5, mode='valid'))
        
        # Check for phase transition
        if cur_mean_avg > self.score_threshold:
            self.last_phase_transition = i
            self.last_high_ml_score = max(np.convolve(self.all_mean_ml_scores, 
                                                     np.ones(5)/5, mode='valid'))
            
            logger.info(f"#### Phase transition detected. High score avg: {self.last_high_ml_score:.3g}, "
                  f"high var: {self.last_high_var:.3g}. ####")
            
            # Update phase transition state
            self.active_phase_transition = True
            self.num_phase_transitions += 1
            
            # Record first phase transition if not already set
            if self.first_phase_transition is None:
                self.first_phase_transition = i
                
            logger.info(f"Will start increasing temperature in {self.patience_phase_trans} iterations, "
                  f"unless a phase reversal is detected first.")
            
            # update the score thresholds if necessary
            if self.adaptive_upper_threshold is not None:
                self.score_threshold = self._get_adaptive_score_threshold()
                logger.info(f"Updated phase transition score threshold to {self.score_threshold:.3g}.")
            
    def detect_phase_transition_reversal(self, i):
        """
        Detect a reversal of a phase transition.
        
        A reversal is detected when the mean score drops below the reversal threshold
        or when the patience period has expired.
        
        Args:
            i (int): Current iteration number
        """
        cur_mean_avg = np.mean(np.array(self.all_mean_ml_scores[-2:]))
        
        # Check for score-based reversal
        if cur_mean_avg < self.reversal_threshold:
            logger.info(f"####### Phase transition reversal detected. ########")
            self.active_phase_transition = False
            
        # Check for patience-based reversal
        if (i > self.last_phase_transition + self.patience):
            logger.info(f"####### Stopping phase transition because we have not detected "
                  f"a phase reversal in {self.patience} iterations. ########")
            self.active_phase_transition = False

        
    def update_temperature(self, i, num_sequences):
        """
        Update the temperature based on the current optimization state.
        
        The temperature is updated based on several factors:
        1. Number of sequences found (to prevent sampling failure)
        2. Simple simulated annealing if enabled
        3. Fixed cool-then-heat cycles if enabled
        4. Adaptive cycling based on phase transitions
        
        Args:
            i (int): Current iteration number
            num_sequences (int): Number of sequences sampled in current iteration
        """
        # Prevent temperature from getting too low (avoid sampling issues)
        if self.T < 0.01:
            self.T *= 1.4
            logger.info(f"Increasing temperature because it got too low, to prevent sampler errors.")
            
        # Increase temperature if too few sequences are being found
        elif num_sequences < np.round(self.n_batch * 0.5):
            if self.T < 10 * self.optimizer_params['T']:
                self.low_temp_threshold = self.T * 1.10
                self.T *= 1.4
                logger.info(f"#### Increased temperature to {self.T:.3g} to find more sequences per iteration. ####")
                logger.info(f"#### Low T threshold set to {self.low_temp_threshold:3g}. ####")
        
        # Simple simulated annealing mode
        elif self.simple_simulated_annealing:
            self.T *= self.cooling_rate
            
        # Fixed cool-then-heat cycling mode
        elif self.cool_then_heat:
            # Initialize cycle state if not already done
            if not hasattr(self, 'heating_phase'):
                self.heating_phase = False  # Start with cooling
                self.cooling_iterations = 0
                self.heating_iterations = 0
                
            if self.heating_phase:
                # Heating phase
                if (self.heating_iterations >= 45) or (self.T >= self.high_temp_threshold):
                    logger.info("### Starting to cool the system. ###")
                    self.heating_phase = False
                    self.cooling_iterations = 1
                    self.T *= self.cooling_rate
                else:
                    self.T /= self.cooling_rate  # Continue heating
                    self.heating_iterations += 1
            else:
                # Cooling phase
                if (self.cooling_iterations >= 45) or (self.T <= self.low_temp_threshold):
                    logger.info(f"### Starting to heat the system. T threshold: {self.low_temp_threshold:3g}. "
                          f"T: {self.T:3g}. ###")
                    self.heating_phase = True
                    self.heating_iterations = 1
                    self.T /= self.cooling_rate
                else:
                    self.T *= self.cooling_rate
                    self.cooling_iterations += 1
        
        # Adaptive temperature cycling based on phase transitions
        else:
            if self.active_phase_transition and (i > self.last_phase_transition + self.patience_phase_trans):
                # Heat to induce phase reversal
                self.T *= self.heating_rate
                logger.info(f"#### Increasing temperature to {self.T:.3g} to achieve a phase reversal. ####")
                
            elif self.last_phase_transition is None:
                # No phase transition detected yet, continue cooling
                self.T *= self.cooling_rate
                
            else:
                # Adjust cooling rate based on number of mutations
                if self.num_mutations < 4:
                    self.T *= np.square(self.cooling_rate) * self.cooling_rate
                else:
                    self.T *= np.square(self.cooling_rate)
        
    def prepare_results(self, n_seqs_to_keep=None):
        """
        Prepare results DataFrames with sequences and optimization statistics.
        
        Args:
            n_seqs_to_keep (int, optional): Number of top sequences to keep. 
                If None, keeps all sequences.
                
        Returns:
            tuple: (df, df_stats)
                - df (pd.DataFrame): Top sequences and their scores
                - df_stats (pd.DataFrame): Optimization statistics per iteration
        """
        if n_seqs_to_keep is None:
            n_seqs_to_keep = len(self.all_scores)
            
        # Sort sequences by score and select top ones
        sorted_indices = np.argsort(self.all_scores)[::-1][:n_seqs_to_keep]
        ranked_sequences = [self.all_sequences[i] for i in sorted_indices]
        ranked_scores = [self.all_scores[i] for i in sorted_indices]
        ranked_counts = [self.seq_iter_counts[i] for i in sorted_indices]
        ranked_first_seen = [self.iter_first_seen[i] for i in sorted_indices]

        # Create sequences DataFrame
        df = pd.DataFrame({
            'sequences': np.array(ranked_sequences),
            'score': np.array(ranked_scores),
            'counts': np.array(ranked_counts, dtype=int),
            'num_mutations': np.array(np.ones(len(ranked_sequences), dtype=int) * self.num_mutations, dtype=int),
            'iteration': np.array(ranked_first_seen, dtype=int)
        })

        # Create statistics DataFrame
        df_stats = pd.DataFrame({
            'iteration': np.array(range(len(self.all_mean_ml_scores)), dtype=int),
            'avg_score': np.array(self.all_mean_ml_scores),
            'var_score': np.array(self.all_variances),
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
        Save optimization results to CSV files.
        
        Args:
            filename (str, optional): Base filename for saving results.
                If None, uses default based on number of mutations.
            n_seqs_to_keep (int, optional): Number of top sequences to keep.
                If None, keeps all sequences.
        """
        if n_seqs_to_keep is not None:
            df, df_stats = self.prepare_results(n_seqs_to_keep)
        else:
            df, df_stats = self.df, self.df_stats
            
        if filename is None:
            filename = f'optimizer_{self.num_mutations}_mutations.csv'
            
        # Save sequences
        seq_filename = 'ranked_sequences_' + filename
        df.to_csv(seq_filename, index=False, float_format='%.5g')
        
        # Save statistics
        stats_filename = filename.replace('ranked_sequences', 'optimizer_stats')
        df_stats.to_csv(stats_filename, index=False, float_format='%.5g')
        
    def plot_scores(self, save_figs=True):
        """
        Generate visualization plots of the optimization process.
        
        Creates three plot groups:
        1. Statistics by iteration (scores, effective samples, temperature)
        2. Score distributions vs temperature
        3. Score density distributions
        
        Args:
            save_figs (bool): Whether to save plots to files
        """
        df, df_stats = self.df, self.df_stats
        df_stats['std_dev_score'] = np.sqrt(df_stats['var_score'])

        sns.set(style="whitegrid")

        # First Group: Statistics by iteration
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

        # Plot avg_score and std_dev_score
        ax1 = sns.lineplot(x='iteration', y='avg_score', data=df_stats, ax=axes[0], 
                          color='b', label='avg score', linewidth=2.5)
        ax2 = ax1.twinx()
        sns.lineplot(x='iteration', y='std_dev_score', data=df_stats, ax=ax2, 
                    color='r', label='score std dev', linewidth=2.5)
        ax1.set_ylabel('Average Score', color=ax1.lines[0].get_color())
        ax2.set_ylabel('Score Std. Dev.', color='r')
        ax1.tick_params(axis='y', labelcolor=ax1.lines[0].get_color())
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Plot n_eff_sites and n_eff_aa
        ax3 = sns.lineplot(x='iteration', y='n_eff_sites', data=df_stats, ax=axes[1], 
                          color='b', label=f'site $n_{{eff}}$', linewidth=2.5)
        ax4 = ax3.twinx()
        sns.lineplot(x='iteration', y='n_eff_aa', data=df_stats, ax=ax4, 
                    color='r', label=f'aa $n_{{eff}}$', linewidth=2.5)
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

        plt.title(f'Optimization with {self.num_mutations} mutations per sequence')

        # Add phase transition markers
        phase_transition_iters = df_stats['iteration'][df_stats['num_phase_transitions'].diff() == 1].values
        for ax in [ax1, ax3, ax5]:
            for pt_iter in phase_transition_iters:
                ax.axvline(x=pt_iter, color='black', linestyle='--', linewidth=1)

        plt.tight_layout()
        if save_figs:
            plt.savefig(f'iteration_statistics_num_mutations_{self.num_mutations}.png', 
                       bbox_inches='tight')
        plt.show()

        # Second Group: Score distributions vs temperature
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Define marker styles for phase transitions
        marker_styles = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'P', 5: '*', 
                        6: 'X', 7: 'v', 8: '<', 9: '>'}
        default_marker = '.'
        num_phase_transition_markers = {
            num: marker_styles.get(num, default_marker) 
            for num in df_stats['num_phase_transitions'].unique()
        }

        # Temperature vs scores
        ax1 = axes[0]
        ax2 = ax1.twinx()
        ax2.grid(False)

        for num_pt, group in df_stats.groupby('num_phase_transitions'):
            marker = num_phase_transition_markers[num_pt]
            sns.scatterplot(x='T', y='avg_score', data=group, ax=ax1, 
                          label='', color='b', s=50, marker=marker, alpha=0.85)
            sns.scatterplot(x='T', y='std_dev_score', data=group, ax=ax2, 
                          label='', color='r', s=50, marker=marker, alpha=0.85)
        ax1.set_ylabel('Average Score', color='b')
        ax2.set_ylabel('Score Std. Dev.', color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.set_xlabel('Temperature')

        # Temperature vs effective samples
        ax3 = axes[1]
        ax4 = ax3.twinx()
        ax4.grid(False)

        for num_pt, group in df_stats.groupby('num_phase_transitions'):
            marker = num_phase_transition_markers[num_pt]
            sns.scatterplot(x='T', y='n_eff_sites', data=group, ax=ax3, 
                          label='', color='b', s=50, marker=marker, alpha=0.85)
            sns.scatterplot(x='T', y='n_eff_aa', data=group, ax=ax4, 
                          label='', color='r', s=50, marker=marker, alpha=0.85)
        ax3.set_ylabel(f'Site $n_{{eff}}$', color='b')
        ax4.set_ylabel(f'AA $n_{{eff}}$', color='r')
        ax3.tick_params(axis='y', labelcolor='b')
        ax4.tick_params(axis='y', labelcolor='r')
        ax3.set_xlabel('Temperature')

        plt.title(f'Temperature relationships with {self.num_mutations} mutations')
        plt.tight_layout()
        if save_figs:
            plt.savefig(f'temperature_scatter_plots_num_mutations_{self.num_mutations}.png', 
                       bbox_inches='tight')
        plt.show()

        # Third Group: Score density distributions
        if self.first_phase_transition is not None:
            median_iteration = self.first_phase_transition
        else:
            median_iteration = df['iteration'].median()

        early_df = df[df['iteration'] <= median_iteration]
        late_df = df[df['iteration'] > median_iteration]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        sns.histplot(early_df['score'], bins=30, stat='density',
                    element='step', color='blue', label='Early Iterations', ax=ax)
        sns.histplot(late_df['score'], bins=30, stat='density',
                    element='step', color='orange', label='Late Iterations', ax=ax)

        plt.axvline(x=self.wt_score, color='black', linestyle='--', 
                   linewidth=2.5, label='WT Score')

        plt.title(f'Score distributions with {self.num_mutations} mutations')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        
        if save_figs:
            plt.savefig(f'score_density_plots_num_mutations_{self.num_mutations}.png', 
                       bbox_inches='tight')
        plt.show()

    def _deduplicate_results(self):
        """
        Deduplicate the results to keep unique sequences and their mean scores.
        
        This method:
        1. Identifies unique sequences
        2. Computes mean scores for duplicates
        3. Tracks first occurrence and count of each sequence
        4. Updates the main results lists with deduplicated data
        """
        unique_sequences = {}       # sequence -> mean score
        sequence_counts = {}        # sequence -> number of occurrences
        first_seen_iteration = {}   # sequence -> first iteration seen
        
        # First pass: Accumulate scores and track iterations
        for sequence, score, iteration in zip(self.all_sequences, 
                                            self.all_scores, 
                                            self.iters_for_seqs):
            if sequence not in unique_sequences:
                # First time seeing this sequence
                unique_sequences[sequence] = score
                sequence_counts[sequence] = 1
                first_seen_iteration[sequence] = iteration
            else:
                # Update running average
                unique_sequences[sequence] += score
                sequence_counts[sequence] += 1
                
        # Second pass: Compute mean scores
        for sequence in unique_sequences:
            unique_sequences[sequence] /= sequence_counts[sequence]
            
        # Update class attributes with deduplicated data
        self.all_sequences = list(unique_sequences.keys())
        self.all_scores = list(unique_sequences.values())
        self.seq_iter_counts = list(sequence_counts.values())
        self.iter_first_seen = list(first_seen_iteration.values())