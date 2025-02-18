import re
import torch
import random
import numpy as np

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY' # 20 canonical amino acids
AA_TO_IDX = {aa:idx for idx, aa in enumerate(AMINO_ACIDS)}
EPS = 1e-10

def generate_single_mutants(ref_seq, sites_to_exclude=[]):
    mutants = []
    L = len(ref_seq)  # Length of the reference sequence

    for i, original_aa in enumerate(ref_seq):
        site = i + 1  # 1-based site index
        if site in sites_to_exclude:
            continue  # Skip sites that are in the exclusion list

        for aa in AMINO_ACIDS:
            if aa != original_aa:  # Only consider mutations
                mutant = f"{original_aa}{site}{aa}"
                mutants.append(mutant)
    return mutants

def pseudolikelihood_ratio(rel_seq, ref_seq, logits):
    # rel_seq is a dictionary {site: aa}, and ref_seq is an absolute sequence in a string. Logits are (L, 20).
    # Using logprobs gives the same answer.
    score = 0.0
    for site, mutated_aa in rel_seq.items():
        site_idx = int(site)  # Already 0-based index
        mutated_aa_idx = AA_TO_IDX[mutated_aa]
        reference_aa_idx = AA_TO_IDX[ref_seq[site_idx]]
        # Compute score differences for the mutated sequence
        logprob_diff_mutant = logits[site_idx, mutated_aa_idx] - logits[site_idx, reference_aa_idx]
        score += logprob_diff_mutant.item()
    return score

def pseudolikelihood_ratio_from_tensor(rel_seq_tensor, ref_seq, logits):
    # rel_seq_tensor is a (N, 2) tensor where the first column is the one-based site index
    # and the second column is the amino acid index. N is number of mutations, with padding of nan's at the end.
    # ref_seq is an absolute sequence string. Logits are (L, 20).
    score = 0.0
    for mutation in rel_seq_tensor: # this loops over the rows
        if torch.isnan(mutation[0]):  # Stop processing when encountering NaN
            break
        site_idx = int(mutation[0].item()) - 1  # Convert to 0-based index
        mutated_aa_idx = int(mutation[1].item())
        reference_aa_idx = AA_TO_IDX[ref_seq[site_idx]]
        # Compute score differences for the mutated sequence
        logprob_diff_mutant = logits[site_idx, mutated_aa_idx] - logits[site_idx, reference_aa_idx]
        score += logprob_diff_mutant.item()
    return score

def compute_num_mutations_from_padded_rel_seqs_tensors(rel_seqs_tensors):
    num_mutations = []
    for rel_seq_tensor in rel_seqs_tensors:
        count = 0
        for mutation in rel_seq_tensor:
            if torch.isnan(mutation[0]):
                break
            count += 1
        num_mutations.append(count)
    return torch.tensor(num_mutations, dtype=torch.float32).unsqueeze(-1)

def pad_rel_seq_tensors_with_nan(rel_seqs_tensors):
    # pad rel seq tensors so that they all have the same size for DataParallel to work correctly
    # Determine the maximum length of sequences in the batch
    max_length = max(tensor.size(0) for tensor in rel_seqs_tensors)

    padded_tensors = []
    for tensor in rel_seqs_tensors:
        pad_size = max_length - tensor.size(0)
        if pad_size > 0:
            # Create padding tensor with NaN values
            padding = torch.full((pad_size, 2), torch.nan, dtype=torch.float32)
            # Concatenate the original tensor with the padding tensor
            padded_tensor = torch.cat((tensor.float(), padding), dim=0)
        else:
            padded_tensor = tensor.float()
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors into a single tensor
    padded_tensors = torch.stack(padded_tensors)

    return padded_tensors

def split_by_sequence(sequences, split_ratio=0.8, ref_seq='NA', return_sequences=False):
    """
    Splits a given list of sequences into training and validation sets, ensuring that the reference sequence
    (identified as `ref_seq`) and a proportion of unique non-reference sequences are included in the training set
    according to the specified split ratio.

    Arguments:
    - sequences: A list of sequences (strings) where the reference sequence name is identified by `ref_seq`.
    - split_ratio (float, optional): The ratio of the dataset to be included in the training set.
      Defaults to 0.8, meaning 80% training and 20% validation.
    - ref_seq (str, optional): The reference sequence name. Defaults to 'NA'.

    Outputs:
    - train_sequences: A list of sequences intended for training, containing a mix of reference and non-reference
      sequences according to the split ratio.
    - val_sequences: A list of sequences intended for validation, containing the remaining sequences not included in
      the training set.

    # Example usage:
    sequences = ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC", "DDD", "AAA"]
    ref_seq = "AAA"
    train_sequences, val_sequences = split_sequences_by_reference(sequences, split_ratio=0.8, ref_seq=ref_seq)

    print("Train sequences:", train_sequences)
    print("Validation sequences:", val_sequences)

    """
    # Step 1: Isolate reference sequence measurements
    ref_seq_indices = [idx for idx, seq in enumerate(sequences) if seq == ref_seq]
    non_ref_seq_indices = [idx for idx, seq in enumerate(sequences) if seq != ref_seq]
    print(f"Found {len(ref_seq_indices)} wildtype sequences in dataset.")
    # Prepare for counting frequencies of non-reference sequences
    sequence_to_indices = {}
    for idx in non_ref_seq_indices:
        seq = sequences[idx]
        if seq not in sequence_to_indices:
            sequence_to_indices[seq] = []
        sequence_to_indices[seq].append(idx)

    n_unique = len(sequence_to_indices) + 1

    # Step 2: Shuffle the list of non-reference unique sequences
    unique_non_ref_sequences = list(sequence_to_indices.keys())
    random.shuffle(unique_non_ref_sequences)

    # Step 3: Allocate sequences to training set based on frequency, starting with the reference sequence
    train_indices = ref_seq_indices
    cnt = 1
    for seq in unique_non_ref_sequences:
        if len(train_indices) / len(sequences) < split_ratio:
            train_indices.extend(sequence_to_indices[seq])
            cnt += 1
        else:
            break

    # Remaining sequences go to validation set
    val_indices = [idx for idx in non_ref_seq_indices if idx not in train_indices]

    print(f"train has {cnt} unique sequences out of {n_unique}.")
    if return_sequences:
        # Extract the sequences using the indices
        train_sequences = [sequences[idx] for idx in train_indices]
        val_sequences = [sequences[idx] for idx in val_indices]
        return train_sequences, val_sequences

    return train_indices, val_indices

def convert_rel_seqs_to_tensors(rel_seqs_dicts):
    # Assumes rel_seqs_dicts has one-based sites, e.g., [{1: F, 87: G}, ...]
    rel_seqs_tensors = []
    for rel_seq_dict in rel_seqs_dicts:
        if rel_seq_dict == {}:
            seq_tensor = torch.empty((0, 2), dtype=torch.long)
        else:
            seq_tensor = torch.tensor([[pos, AA_TO_IDX[aa]] for pos, aa in rel_seq_dict.items()], dtype=torch.long)
        rel_seqs_tensors.append(seq_tensor)
    return rel_seqs_tensors

def rel_sequences_to_dict(rel_sequences, sep='-'):
    """
    Converts a list of relative sequence strings to a nested dictionary.
    :param rel_sequences: List of strings, each containing concatenated mutations.
    :param sep: Separator used between mutations in the relative sequences.
    :return: A nested dictionary where the first key is the zero-based index of the sequences,
             and the second level key is the one-based mutation site, with the value being the mutated amino acid.
    """
    sequences_dict = {}
    for idx, rel_seq in enumerate(rel_sequences):
        sequence_mutations = {}
        mutations = rel_seq.split(sep)
        for mutation in mutations:
            if mutation:  # Non-empty mutation string
                #print(mutation)
                original_aa, site, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
                sequence_mutations[site] = mutated_aa
        sequences_dict[idx] = sequence_mutations
    return sequences_dict

def create_absolute_sequences_list(relative_sequences, reference_sequence, separator='-'):
    """
    Create a list of absolute sequences from a list of relative sequences and a reference sequence.
    This function supports input in both string format ("A1Y-N5Q-P10Q-Y15T") and dictionary format ({1: 'Y', 5: 'Q', 10: 'Q', 15: 'T'}).
    It maintains duplicates, unlike a version that outputs a dictionary.

    Parameters:
    relative_sequences (list): A list containing relative sequences either as strings or as dictionaries.
    reference_sequence (str): The sequence to be used as reference for applying mutations.
    separator (str): The character used to separate mutations in the string representation of relative sequences.
                     Default is '-'.

    Returns:
    list: A list of absolute sequences, preserving the order and duplicates from the input.

    Example Usage:
    reference_seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    relative_seqs = ["A1Y-N5Q-P10Q-Y15T", {"3": "W", "20": "R"}]
    absolute_seqs_list = create_absolute_sequences_list(relative_seqs, reference_seq)
    print(absolute_seqs_list)
    """
    absolute_sequences = []

    for rel_seq in relative_sequences:
        # Initialize the absolute sequence as a list for efficient mutation
        absolute_seq = list(reference_sequence)

        if rel_seq == 'NA' or rel_seq == '':  # Special case where 'NA' means use the reference sequence without changes
            pass
        elif isinstance(rel_seq, str):
            # The relative sequence is in string format: split and apply each mutation
            mutations = [mutation.strip() for mutation in rel_seq.split(separator)]
            for mutation in mutations:
                original_aa, site, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
                absolute_seq[site - 1] = mutated_aa  # Apply mutation
        elif isinstance(rel_seq, dict):
            # The relative sequence is in dictionary format: apply each mutation
            for site, mutated_aa in rel_seq.items():
                site = int(site)  # Convert site to integer
                absolute_seq[site - 1] = mutated_aa  # Apply mutation
        else:
            # Unsupported format, raise an error
            raise ValueError("Unsupported relative sequence format")

        # Join the list back into a string and add to the output list
        absolute_sequences.append(''.join(absolute_seq))

    return absolute_sequences

def apply_mutations(reference_sequence, mutations_dict):
    """
    Applies the specified mutations to the reference sequence.

    Args:
        reference_sequence (str): The original amino acid sequence.
        mutations_dict (dict): A dictionary where keys are the positions (1-based indexing) of mutations
                               and values are the mutated amino acids.

    Returns:
        str: The mutated amino acid sequence.
    """
    if mutations_dict == {}:
        return reference_sequence # no mutations
    # Convert the reference sequence to a list of characters for easy manipulation
    mutated_sequence = list(reference_sequence)
    # Iterate through the mutations and apply each one to the sequence
    for position, new_aa in mutations_dict.items():
        # Convert the 1-based indexing used in mutations_dict to 0-based indexing used in lists
        index = position - 1
        # Replace the amino acid at the specified position with the new one
        mutated_sequence[index] = new_aa
    # Convert the list back into a string and return

    return ''.join(mutated_sequence)

def compute_Neff_for_probmatrix(p):
    def compute_Neff(sorted_p):
        n = len(sorted_p)
        x = np.sum(sorted_p * np.arange(1, n + 1))
        Neff = 2 * x - 1
        return Neff

    if len(p.shape) == 1:
        # p is a vector
        sorted_p = np.sort(p)[::-1]  # Sort in descending order
        Neff = compute_Neff(sorted_p)
        return {"Neff": Neff}
    elif len(p.shape) == 2:
        # p is a matrix (joint probability distribution)

        # Flatten the matrix and compute Neff
        flattened_p = p.flatten()
        sorted_flattened_p = np.sort(flattened_p)[::-1]
        Neff = compute_Neff(sorted_flattened_p)

        # Compute Neff for the marginal distribution of columns
        marginal_cols = np.sum(p, axis=0)
        sorted_marginal_cols = np.sort(marginal_cols)[::-1]
        Neff_cols = compute_Neff(sorted_marginal_cols)

        # Compute Neff for the marginal distribution of rows
        marginal_rows = np.sum(p, axis=1)
        sorted_marginal_rows = np.sort(marginal_rows)[::-1]
        Neff_rows = compute_Neff(sorted_marginal_rows)

        return {"Neff": Neff, "Neff_cols": Neff_cols, "Neff_rows": Neff_rows}
    else:
        raise ValueError("Input p must be a 1D or 2D numpy array.")

def create_score_matrix(reference_sequence, single_mutants, ml_scores, wt_scores = 0.0):
    """
    Create a score matrix from single mutant strings and their corresponding ML scores.

    Args:
        reference_sequence (str): Wild-type sequence.
        single_mutants (list[str]): List of single mutants in the format 'A456Y'.
        ml_scores (np.ndarray): Array of ML scores corresponding to the single mutants.
        wt_scores (float): the default value of wild type scores.

    Returns:
        np.ndarray: Score matrix of shape (20, L), where L is the length of the reference sequence.
    Note that because only single mutants are found, the wildtype scores are set to the default of zero.
    """
    # Initialize score matrix
    L = len(reference_sequence)
    score_matrix = np.ones((20, L))*wt_scores

    # Fill the score matrix
    for mutant, score in zip(single_mutants, ml_scores):
        wt_aa = mutant[0]
        site = int(mutant[1:-1]) - 1  # Convert to 0-based index
        mut_aa = mutant[-1]
        #wt_idx = AA_TO_IDX[wt_aa]
        mut_idx = AA_TO_IDX[mut_aa]
        score_matrix[mut_idx, site] = score

    return score_matrix
