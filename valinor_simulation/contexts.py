import numpy as np


def getContextMatrix(
    nCellLines=30, nGenes=100, nContexts=10, nVariantFrac=0.05, variance=0.1
):
    linesPerContext = nCellLines // nContexts
    nVariantGenes = int(nVariantFrac * nGenes)
    contexts = np.zeros((nContexts, nGenes))

    positions = np.random.randint(0, nGenes, (nContexts, nVariantGenes))

    factor = np.random.normal(0, np.sqrt(variance), size=(nContexts, nVariantGenes))

    for c in range(nContexts):
        contexts[c, positions[c]] = factor[c]

    mat = np.repeat(contexts, [linesPerContext] * nContexts, axis=0)

    return mat


def generate_context_matrices(
    num_genes, num_contexts, fraction_gene_pairs_in_context, scale=0.1
):
    """
    Generate context matrices for gene interactions.

    Parameters:
    - num_genes: Number of genes.
    - num_contexts: Number of context-specific groups of cell lines.
    - fraction_gene_pairs_in_context: Fraction of gene pairs that are present in a context.

    Returns:
    - A list of context matrices for each context.
    """

    # Calculate the number of gene pairs that should be present in a context
    num_gene_pairs_to_modify = int(
        num_genes * (num_genes - 1) * fraction_gene_pairs_in_context / 2
    )

    context_matrices = []

    for _ in range(num_contexts):
        # Start with a matrix of zeros (indicating no change)
        context_matrix = np.zeros((num_genes, num_genes))

        # Randomly select gene pairs to modify, ensuring we don't select diagonal pairs
        gene_pairs_to_modify = []
        while len(gene_pairs_to_modify) < num_gene_pairs_to_modify:
            pair = np.random.choice(num_genes, size=2, replace=False)
            if pair[0] != pair[1]:
                gene_pairs_to_modify.append(pair)

        for pair in gene_pairs_to_modify:
            # Generate a random multiplier between 0 and scale for the interaction
            # multiplier = np.random.uniform(0, scale)

            multiplier = np.random.normal(scale, scale / 5)
            multiplier *= np.random.choice([1, -1])

            context_matrix[pair[0], pair[1]] = multiplier
            context_matrix[pair[1], pair[0]] = multiplier

        context_matrices.append(context_matrix)

    return context_matrices


def assign_contexts_to_cell_lines(
    total_cell_lines, num_contexts, unique_contexts=False
):
    """
    Assign contexts to cell lines. Each cell line can have multiple contexts, or each context can be unique to a cell line.

    Parameters:
    - total_cell_lines: Total number of cell lines.
    - num_contexts: Number of context-specific groups of cell lines.
    - unique_contexts: If True, each context is assigned to exactly one cell line.

    Returns:
    - A dictionary mapping each cell line to its associated context indices.
    """

    cell_line_to_contexts = {}

    if unique_contexts:
        # Ensure that the number of contexts is not greater than the number of cell lines
        if num_contexts > total_cell_lines:
            raise ValueError(
                "Number of contexts cannot be greater than the number of cell lines for unique assignment."
            )

        # Shuffle the contexts and assign each to a different cell line
        contexts = np.random.permutation(num_contexts)
        for i in range(total_cell_lines):
            context_index = contexts[i % num_contexts]
            cell_line_to_contexts[i] = [context_index]
    else:
        for i in range(total_cell_lines):
            # Randomly assign one or more contexts to each cell line
            assigned_contexts = np.random.choice(
                num_contexts, size=np.random.randint(1, num_contexts + 1), replace=False
            )
            cell_line_to_contexts[i] = assigned_contexts

    return cell_line_to_contexts
