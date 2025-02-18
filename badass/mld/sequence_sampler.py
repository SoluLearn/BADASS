import numpy as np
from badass.utils.sequence_utils import AMINO_ACIDS, AA_TO_IDX, EPS, compute_Neff_for_probmatrix

class CombinatorialMutationSampler:
    """
    A class for sampling protein sequences based on mutation probability distributions.

    This sampler allows for flexible initialization and updating of mutation probabilities using
    various methods, including random uniform sampling, provided probability matrices, or Boltzmann
    distributions derived from score matrices.

    Attributes:
        sequence (str): The reference sequence (wild-type) of the protein.
        L (int): The length of the reference sequence.
        forbidden_sites (list): A list of 1-based site indices where mutations are not allowed.
        temperature (float): The temperature parameter used in Boltzmann sampling.
        q (np.ndarray): The flattened probability distribution used for sampling mutations.
        sampling (str): The mode of sampling ('random', 'pmf', 'boltzmann').
        prob_matrix (np.ndarray): The 2D probability matrix used to generate q.
        mutation_idx_to_str_mapping (list): A list of tuples mapping each index in q to a (site, amino_acid) pair.

    Methods:
        update_boltzmann_distribution(new_temperature=None, new_score_matrix=None):
            Updates the Boltzmann probability matrix with a new temperature or score matrix.

        update_probability_matrix(new_probability_matrix):
            Updates the probability matrix directly with a new provided matrix.
    """
    def __init__(self, sequence, forbidden_sites=None, temperature=1.0, sampling="random",
                 probability_matrix=None, score_matrix=None, verbose=False):
        self.sequence = sequence
        self.L = len(sequence)
        self.forbidden_sites = forbidden_sites or []
        self.temperature = temperature
        self.q = None  # Placeholder for the sampling distribution over mutations
        self.sampling = sampling
        self.verbose = verbose
        if sampling == "random":
            prob_matrix = self._set_random_probability_matrix()
        elif sampling == "pmf":
            prob_matrix = self._set_probability_matrix(probability_matrix)
        elif sampling == "boltzmann":
            self.score_matrix = score_matrix
            prob_matrix = self._set_boltzmann_probability_matrix()
        # Renormalize the matrix
        self.prob_matrix = prob_matrix / prob_matrix.sum()
        #self._compute_marginals()
        self.q =  self.prob_matrix.flatten()
        self._create_mutation_mapping()

    def _compute_marginals(self):
        """Compute marginal probabilities for sites and amino acids.
        self.p_sites: probability of each site summing to 1.0
        self.P_aa: probability of each amino acid at each site, summing to 1.0 at each site"""

        # Marginals for sites
        self.p_sites = np.sum(self.prob_matrix, axis=0)  # Sum over amino acids to get a value per site
        self.p_sites = self.p_sites / np.sum(self.p_sites)  # Normalize exactly

        # Initialize P_aa with the same shape as joint_probabilities
        self.p_aa = np.ones_like(self.prob_matrix)*(1/20)

        # Iterate over each site
        for j in range(self.p_sites.shape[0]):
            if self.p_sites[j] > 0:
                # Normalize the column
                self.p_aa[:, j] = self.prob_matrix[:, j] / self.p_sites[j]
        # Ensure the columns are normalized
        self.p_aa = self.p_aa / self.p_aa.sum(axis=0, keepdims=True)

    def _zero_out_forbidden_sites(self, matrix):
        """Zero out the probabilities at forbidden sites in probability matrix."""
        for site in self.forbidden_sites:
            site_idx = site - 1  # Convert to 0-based index
            matrix[:, site_idx] = 0.0
        return matrix

    def _zero_out_wildtype_amino_acids(self, matrix):
        """Zero out the probabilities for wild-type amino acids in probability matrix."""
        for i, aa in enumerate(self.sequence):
            aa_idx = AA_TO_IDX[aa]
            matrix[aa_idx, i] = 0.0
        return matrix

    def _create_mutation_mapping(self):
        """Create a mapping from flattened probability matrix indices to (site, amino acid) tuples."""
        self.mutation_idx_to_str_mapping = []  # This needs to match the order of q obtained by .flatten() on a matrix (20, L)
        # .flatten() concantenates rows, so need to loop over sites first here.a
        for aa_idx in AA_TO_IDX.values():
            for site in range(self.L):
                self.mutation_idx_to_str_mapping.append((site + 1, aa_idx))

    def _set_random_probability_matrix(self):
        """Initialize a uniform distribution over allowed mutations."""
        prob_matrix = np.ones((20, self.L)) / 19.0  # Uniform over non-wild-type amino acids
        prob_matrix = self._zero_out_forbidden_sites(prob_matrix)
        prob_matrix = self._zero_out_wildtype_amino_acids(prob_matrix)
        return prob_matrix

    def _set_probability_matrix(self, prob_matrix):
        """Set a provided probability matrix and handle forbidden sites."""
        assert prob_matrix.shape == (20, self.L), "Probability matrix must be of shape (20, L)"
        # Zero out forbidden sites and wild-type amino acids
        prob_matrix = prob_matrix.copy()
        prob_matrix = self._zero_out_forbidden_sites(prob_matrix)
        prob_matrix = self._zero_out_wildtype_amino_acids(prob_matrix)
        return prob_matrix

    def _set_boltzmann_probability_matrix(self, temperature=None):
        """Set a provided score matrix and apply the Boltzmann transformation to obtain q."""
        assert self.score_matrix.shape == (20, self.L), "Score matrix must be of shape (20, L)"
        self.temperature = temperature if temperature else self.temperature
        # Clip scores to avoid overflow
        score_matrix = np.clip(self.score_matrix, a_min=None, a_max=45)
        prob_matrix = np.exp(score_matrix / (self.temperature + np.finfo(float).eps))
        #prob_matrix = np.exp(np.clip(self.score_matrix / (self.temperature + np.finfo(float).eps), a_min=None, a_max=1000))
        prob_matrix = self._zero_out_forbidden_sites(prob_matrix)
        prob_matrix = self._zero_out_wildtype_amino_acids(prob_matrix)
        return prob_matrix

    def update_boltzmann_distribution(self, new_temperature=None, new_score_matrix=None):
        """Update the Boltzmann probability matrix."""
        if new_temperature is None and new_score_matrix is None:
            raise ValueError("Either a new temperature or score matrix must be provided to update Boltzmann distribution.")
        if new_temperature is not None:
            self.temperature = new_temperature
        if new_score_matrix is not None:
            assert new_score_matrix.shape == (20, self.L), "New score matrix must be of shape (20, L)"
            self.score_matrix = new_score_matrix
        prob_matrix = self._set_boltzmann_probability_matrix()
        self.prob_matrix = prob_matrix / prob_matrix.sum()
        #self._compute_marginals()
        self.q = self.prob_matrix.flatten()

    def update_probability_matrix(self, new_probability_matrix):
        """Update the probability matrix directly."""
        assert new_probability_matrix.shape == (20, self.L), "New probability matrix must be of shape (20, L)."
        prob_matrix = self._zero_out_forbidden_sites(new_probability_matrix)
        prob_matrix = self._zero_out_wildtype_amino_acids(prob_matrix)
        self.prob_matrix = prob_matrix / prob_matrix.sum()
        #self._compute_marginals()
        self.q = self.prob_matrix.flatten()

    def sample_single_mutants(self, library_size, dedupe=False, replace=False):
        """
        Sample single mutants based on the probability distribution q.

        Parameters:
            library_size (int): The number of mutants to sample.
            dedupe (bool): Whether to deduplicate the sampled mutants if sampling with replacement.

        Returns:
            list: A list of mutation strings in the format 'A1C', 'G4T', etc.
        """
        # Case 1: Direct return of all mutants if library size matches non-zero entries

        # Identify non-zero entries in q
        non_zero_indices = np.where(self.q > 0)[0]
        num_non_zero = len(non_zero_indices)
        print(f"There's {num_non_zero} entries in q.")
        if library_size == num_non_zero and replace is False:
            return [self._mutation_to_string(idx) for idx in non_zero_indices]

        # Sample from q according to its probabilities
        if (num_non_zero < library_size) and replace is False:
            if self.verbose:
                print(f"There's {num_non_zero} non-zero probability mutations, so sampling all unique outcomes only.")
            sampled_indices = non_zero_indices
        else:
            sampled_indices = np.random.choice(len(self.q), size=library_size, replace=replace, p=self.q)

        if dedupe:
            sampled_indices = np.unique(sampled_indices)
            #while len(sampled_indices) < library_size:
            #    additional_samples = np.random.choice(len(self.q), size=library_size - len(sampled_indices), replace=True, p=self.q)
            #    sampled_indices = np.unique(np.concatenate((sampled_indices, additional_samples)))

        # Convert sampled indices to mutation strings
        mutants = [self._mutation_to_string(idx) for idx in sampled_indices]
        return mutants

    def sample_multi_mutants(self, num_mutations, library_size, discard_bad_sequences=True, dedupe=False):
        """
        Sample sequences with a specified number of mutations.

        Parameters:
            num_mutations (int): The number of mutations desired per sequence.
            library_size (int): The number of sequences to sample.
            discard_bad_sequences (bool): Whether to discard sequences with fewer than num_mutations due to site deduplication.
            dedupe (bool): Whether to deduplicate the final sequences.

        Returns:
            list: A list of sequences with the specified number of mutations.
        """
        retry_limit = 10
        retry_count = 0
        min_acceptable_sequences = int(0.9 * library_size)

        while retry_count < retry_limit:
            # Determine the total number of mutations to sample
            alpha = 1.0
            if ((dedupe is True) and (discard_bad_sequences is True)):
                alpha = 2.0
            elif ((dedupe is True) and (discard_bad_sequences is False)) or ((dedupe is False) and (discard_bad_sequences is True)):
                alpha = 1.5
            total_mutations_to_sample = int(np.round(alpha * num_mutations * library_size))

            # Sample all mutations at once based on the probability distribution q
            sampled_indices = np.random.choice(len(self.q), size=total_mutations_to_sample, replace=True, p=self.q)

            # Dictionary to track sequences and their counts
            sequences_dict = {}

            i = 0 # index of sampled mutation being processed
            while len(sequences_dict) < library_size and i < total_mutations_to_sample - num_mutations + 1:
                mutation_dict = {}
                mutation_set = set()
                for _ in range(num_mutations):
                    if i >= total_mutations_to_sample:
                        break
                    index = sampled_indices[i]
                    site, aa_idx = self.mutation_idx_to_str_mapping[index]

                    # Handle site duplicates, keep trying to find mutations on sites not present on sequence yet
                    if discard_bad_sequences:
                        while site in mutation_set:
                            i += 1
                            if i >= total_mutations_to_sample:
                                break
                            index = sampled_indices[i]
                            site, aa_idx = self.mutation_idx_to_str_mapping[index]

                    mutation_dict[site] = index # this will just choose the last mutation on repeated sites when keeping all sequences
                    mutation_set.add(site)
                    i += 1

                if discard_bad_sequences and len(mutation_dict) < num_mutations:
                    continue

                # Convert the mutation dictionary to a sorted mutation string
                sorted_mutations = sorted(mutation_dict.items())
                mutation_string = '-'.join([self._mutation_to_string(idx) for _, idx in sorted_mutations])

                # Store the sequence in the dictionary, incrementing the count if it already exists
                if mutation_string in sequences_dict:
                    sequences_dict[mutation_string] += 1
                else:
                    sequences_dict[mutation_string] = 1

            # Handle deduplication or maintaining counts
            if dedupe:
                sequences = list(sequences_dict.keys())
            else:
                sequences = []
                for seq, count in sequences_dict.items():
                    sequences.extend([seq] * count)

            # Check if the number of sequences is acceptable
            if len(sequences) >= min_acceptable_sequences:
                break

            # If not, increment the retry count and try again
            retry_count += 1

        # Ensure we return exactly library_size sequences if possible, otherwise return as many as we have
        if len(sequences) > library_size:
            sequences = np.random.choice(sequences, size=library_size, replace=False).tolist()

        return sequences

    def sample_mutant_library(self, library_size, mutation_proportions, discard_bad_sequences=False, dedupe=False):
        """
        Sample a library of sequences with varying numbers of mutations based on specified proportions.

        Parameters:
            library_size (int): The total number of sequences to sample.
            mutation_proportions (list): A list of non-negative numbers indicating the proportion of sequences to have 1, 2, 3, ... mutations.
            discard_bad_sequences (bool): Whether to discard sequences with fewer than the specified number of mutations.
            dedupe (bool): Whether to deduplicate the final sequences.

        Returns:
            list: A combined list of sequences according to the specified proportions.
        """
        # Normalize the mutation proportions
        mutation_proportions = np.array(mutation_proportions)
        mutation_proportions = mutation_proportions / mutation_proportions.sum()

        # Determine the number of sequences to generate for each mutation count
        sequence_counts = np.ceil(mutation_proportions * library_size).astype(int)

        # Adjust the last element to ensure the total number of sequences equals library_size
        sequence_counts[-1] += library_size - sequence_counts.sum()

        # Generate sequences for each mutation count
        all_sequences = []
        for num_mutations, count in enumerate(sequence_counts, start=1):
            if count > 0:
                if num_mutations == 1:
                    sequences = self.sample_single_mutants(count, dedupe=dedupe, replace=not dedupe)
                else:
                    sequences = self.sample_multi_mutants(num_mutations, count, discard_bad_sequences, dedupe)
                all_sequences.extend(sequences)

        # Ensure we return exactly library_size sequences
        if len(all_sequences) > library_size:
            all_sequences = np.random.choice(all_sequences, size=library_size, replace=False).tolist()

        return all_sequences

    def _mutation_to_string(self, index):
        """
        Convert an index in q to a mutation string.

        Parameters:
            index (int): Index in the flattened probability array q.

        Returns:
            str: A string representing the mutation, e.g., 'A1C'.
        """
        site, aa_idx = self.mutation_idx_to_str_mapping[index]
        wt_amino = self.sequence[site - 1]
        mutant_amino = AMINO_ACIDS[aa_idx]
        return f"{wt_amino}{site}{mutant_amino}"
