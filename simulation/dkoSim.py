import argparse

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

import numpy as np

import random
import string

import seaborn as sns

colours = sns.color_palette()

from pprint import pprint

from tqdm import tqdm

import pandas as pd

import copy

import functools

import time

import pickle

# Extend the ACE parameterisation to double KO

# Mutation matrix -> (mutation, essentiality factor)
# or cell type, tissue type, etc

# [[1, 1, ..., 2.0, 1, ..., -0.3, ...],
#  [1, 1, ..., -0.2, 1, ..., 1, ...]]


def genePairStr_(g1, g2):
    return str(g1) + "~" + str(g2)


@functools.lru_cache(maxsize=None)
def genePairStr(g1, g2):
    if g2 > g1:
        return genePairStr_(g1, g2)
    else:
        return genePairStr_(g2, g1)


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


def generate_context_matrices(num_genes, num_contexts, fraction_gene_pairs_in_context, scale = 0.1):
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

            multiplier = np.random.normal(scale, scale / 2)
            multiplier *= np.random.choice([1, -1])

            context_matrix[pair[0], pair[1]] = multiplier
            context_matrix[pair[1], pair[0]] = multiplier

        context_matrices.append(context_matrix)

    return context_matrices


def assign_contexts_to_cell_lines(total_cell_lines, num_contexts, unique_contexts=False):
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
            raise ValueError("Number of contexts cannot be greater than the number of cell lines for unique assignment.")

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


def negativeBinomial(mean, variance=None, size=None):
    # Minimum observable
    mean[mean < 1e-8] = 10.0

    if variance is None:
        variance = 1.5 * mean

    n_nb = -(mean ** 2 / (mean - variance))
    p_nb = 1.0 - (mean / (variance + 1e-8))

    p_nb = np.clip(p_nb, 0, 1)

    return np.random.negative_binomial(np.maximum(1e-4, n_nb), 1.0 - p_nb, size=size)


def genCellLine(
    prototypeDKO,
    resampleFrac=0.2,
    fluctuateStd=0.01,
    context=None,
    gi_contexts=None,
    gi_context_lists=None,
):
    # Simplest: Fluctuate a fraction of essentialities, re-generate the rest to simulate context differences
    # Fluctuate efficiencies

    # More complicated: Re-generate the rest according to 'driver' mutation matrix

    nGenes = prototypeDKO.nGenes
    resampleIdx = np.random.choice(
        range(nGenes), int(resampleFrac * nGenes), replace=False
    )

    newGeneEss = prototypeDKO.geneEssentiality + np.random.normal(
        0, fluctuateStd, len(prototypeDKO.geneEssentiality)
    )
    newGeneEss[resampleIdx] = np.random.normal(0.1, 0.2, size=len(resampleIdx))

    # If we're adding more context, don't include the prototype context GIs!

    if not (gi_contexts == None):

        newSyn = np.random.normal(
            0, fluctuateStd, size = prototypeDKO.synergies.shape
        )

    else:

        newSyn = prototypeDKO.synergies + np.random.normal(
            0, fluctuateStd, len(prototypeDKO.synergies)
        )


        newSyn[resampleIdx, :][:, resampleIdx] = np.random.normal(
            0.0, 0.1, size=(len(resampleIdx), len(resampleIdx))
        )

    newSyn = np.tril(newSyn) + np.tril(newSyn, -1).T

    if np.random.randint(0, 2) == 1:
        # if 1:

        newRNAEfficiencies = np.clip(
            prototypeDKO.sgRNAEfficiencies
            + np.random.normal(0, 0.01, len(prototypeDKO.sgRNAEfficiencies)),
            0,
            1,
        )

    else:
        sgRNAEfficiencies1 = np.clip(
            np.random.normal(0.65, 0.02, size=nGenes * 1),
            0,
            1,
        )

        sgRNAEfficiencies2 = np.clip(
            np.random.normal(0.95, 0.02, size=nGenes * 1),
            0,
            1,
        )
        newRNAEfficiencies = np.vstack(
            (sgRNAEfficiencies1, sgRNAEfficiencies2)
        ).reshape((-1,), order="F")

    newPairEfficiency = np.clip(
        prototypeDKO.pairEfficiency
        + np.random.normal(0, 0.01, len(prototypeDKO.pairEfficiency)),
        0,
        1,
    )

    newDKO = DoubleKO(
        nGenes=prototypeDKO.nGenes,
        context=context,
        geneEssentiality=newGeneEss,
        synergies=newSyn,
        sgRNAEfficiencies=newRNAEfficiencies,
        pairEfficiency=newPairEfficiency,
        gi_contexts=gi_contexts,
        gi_context_lists=gi_context_lists,
    )

    return newDKO


def addCellLines(
    prototypeDKO,
    dfDKO,
    nCellLines,
    contexts=None,
    gi_contexts=None,
    gi_context_lists=None,
    resampleFrac=0.2,
    fluctuateStd=0.01,
    offsets=None,
    returnCounts=False,
):
    dfDKO["cell_line"] = 0

    dfs = [dfDKO]
    dkos = [prototypeDKO]

    for i in tqdm(range(1, nCellLines)):
        dko = genCellLine(
            prototypeDKO,
            resampleFrac,
            fluctuateStd,
            context=contexts[i] if not contexts is None else None,
            gi_contexts=gi_contexts,
            gi_context_lists=gi_context_lists[i],
        )
        df = populateCombinationDF(dko, returnCounts=returnCounts)

        if not (offsets is None):
            if not returnCounts:
                # Just offset according to the 'true' value as we don't have to
                # correct for essentiality ~0 but not == 0
                df["value"] += offsets[i]
            else:
                # Convert counts to LFC
                countsOffset = (
                    np.power(2, offsets[i]) * dko.nInitialCells - dko.nInitialCells
                )
                df["value"] += countsOffset[i]

        df["g1_idx"] += len(np.unique(df["g1_idx"])) * i
        df["g2_idx"] += len(np.unique(df["g2_idx"])) * i

        if "gene_pair_index" in df.columns:
            df["gene_pair_index"] += len(np.unique(df["gene_pair_index"])) * i

        df["cell_line"] = i

        dfs.append(df)
        dkos.append(dko)

    newDF = pd.concat(dfs)

    return newDF, dkos


def populateCombinationDF(dko, returnCounts=False):
    if returnCounts:
        final_counts, sgRNAEssentialities, init_counts, lfcs = dko.getLFCs(
            returnCounts=returnCounts
        )
    else:
        lfcs, sgRNAEssentialities = dko.getLFCs(returnCounts=returnCounts)

    # dko.plot(lfcs, sgRNAEssentialities)

    geneIdx = dko.rnaIdx.reshape(-1, 2)

    # LFC gene indices

    gene1 = geneIdx[:, 0]
    gene2 = geneIdx[:, 1]

    ess1 = dko.geneEssentiality[gene1]
    ess2 = dko.geneEssentiality[gene2]

    syn = dko.synergies[dko.rnaIdx[:, :, 0], dko.rnaIdx[:, :, 1]].T.flatten()

    # LFC RNA indices

    rnas = np.array(range(0, dko.nGenes * dko.nGuidesPerGene))
    rnaIdx = np.dstack(np.meshgrid(rnas, rnas)).reshape(-1, 2)

    rna1 = rnaIdx[:, 0]
    rna2 = rnaIdx[:, 1]

    # LFC pair indices
    # This, remarkably, is a real bottleneck :(

    gene_pairs = [genePairStr(g1, g2) for g1, g2 in geneIdx]
    guide_pairs = [genePairStr(g1, g2) for g1, g2 in rnaIdx]

    # Remove pairs where g1 = g2

    sameGenes = np.array(
        [p.split("~")[0] == p.split("~")[1] for p in gene_pairs], dtype=bool
    )

    gene_pairs = np.array(gene_pairs)[sameGenes == False]
    guide_pairs = np.array(guide_pairs)[sameGenes == False]

    unique_gene_pairs = list(np.unique(gene_pairs))
    unique_guide_pairs = list(np.unique(guide_pairs))

    unique_gene_pairs = {unique_gene_pairs[i]: i for i in range(len(unique_gene_pairs))}
    unique_guide_pairs = {
        unique_guide_pairs[i]: i for i in range(len(unique_guide_pairs))
    }

    gene_pair_index = [unique_gene_pairs[g] for g in gene_pairs]
    guide_pair_index = [unique_guide_pairs[g] for g in guide_pairs]

    dfCombs = pd.DataFrame(
        {
            "value": lfcs if not returnCounts else final_counts,
            "plasmid": lfcs if not returnCounts else init_counts,
            "same_genes": sameGenes,
            "g1_idx": gene1,
            "g2_idx": gene2,
            "gene1": gene1,
            "gene2": gene2,
            "guide1_index": rna1,
            "guide2_index": rna2,
            "ess1": ess1,
            "ess2": ess2,
            "syn": syn,
        }
    )

    dfCombs = dfCombs.query("same_genes == False").copy()
    dfCombs["gene_pair_index"] = gene_pair_index
    dfCombs["guide_pair_index"] = guide_pair_index
    dfCombs["GuidePair"] = guide_pair_index

    return dfCombs


def addReplicates(df, nReplicates, returnCounts=False):
    copies = []
    for i in tqdm(range(nReplicates)):
        replicate = df.copy()

        if not returnCounts:
            std = np.sqrt(1.0 / np.random.gamma(2, 5, size=len(replicate)))

            replicate["value"] = np.random.normal(replicate["value"].values, std)

        else:
            # replicate["value"] = np.random.poisson(replicate["value"])
            replicate["value"] = negativeBinomial(
                replicate["value"].values, 10.0 * replicate["value"].values
            )

        copies.append(replicate)

    newDF = pd.concat(copies)

    return newDF


def makeSingletonsDF(dkos, offsets=None, returnCounts=False):
    # TO DO: Use a separate make cell lines function, like for combinations

    dfs = []

    for i, dko in tqdm(enumerate(dkos)):
        singletonDKO, values = makeSingletons(dko, returnCounts=returnCounts)

        if returnCounts:
            (
                lfcs_s,
                initial_counts_s,
                final_counts_s,
                sgRNAEssentialities_s,
                rna1_s,
                gene1_s,
                rna2_s,
                gene2_s,
            ) = values
        else:
            lfcs_s, sgRNAEssentialities_s, rna1_s, gene1_s, rna2_s, gene2_s = values

        if not (offsets is None):
            if not returnCounts:
                # Just offset according to the 'true' value as we don't have to
                # correct for essentiality ~0 but not == 0
                lfcs_s += offsets[i]
            else:
                # Convert counts to LFC
                countsOffset = (
                    np.power(2, offsets[i]) * singletonDKO.nInitialCells
                    - singletonDKO.nInitialCells
                )
                lfcs_s += countsOffset[i]

        gene1_s_idx = gene1_s + len(np.unique(gene1_s)) * i
        gene2_s_idx = gene2_s + len(np.unique(gene2_s)) * i

        guide_pair_index = np.array(range(len(gene1_s_idx)))

        dfSgl = pd.DataFrame(
            {
                "value": lfcs_s if not returnCounts else final_counts_s,
                "plasmid": lfcs_s if not returnCounts else initial_counts_s,
                "g1_idx": gene1_s_idx,
                "g2_idx": gene2_s_idx,
                "gene1": gene1_s,
                "gene2": gene2_s,
                "guide1_index_s": rna1_s,
                "guide2_index_s": rna2_s,
                "guide_pair_index": guide_pair_index,
                # The same as guide_idx + cell line offset
                "gene_pair_index": guide_pair_index + len(guide_pair_index) * i,
                "ess1": sgRNAEssentialities_s.ravel(),
                "cell_line": i,
            }
        )

        dfs.append(dfSgl)

    newDF = pd.concat(dfs)

    return newDF


def makeSingletons(dko, returnCounts=False):
    singletonDKO = copy.copy(dko)

    singletonDKO.synergies = np.zeros((singletonDKO.nGenes, 3))

    singletonDKO.essentialities = singletonDKO.combinedEssentiality(
        singletonDKO.geneEssentiality, singletons=True
    )

    singletonDKO.efficiencies = singletonDKO.combinedEfficiencies(
        singletonDKO.sgRNAEfficiencies, singletons=True
    )

    # Efficiency for guide pair term, sample separately for
    # singleton - NE combinations
    singletonDKO.pairEfficiency = np.clip(
        np.random.normal(
            0.95,
            0.02,
            size=(
                singletonDKO.nGenes * singletonDKO.nGuidesPerGene,
                3 * singletonDKO.nGuidesPerGene,
            ),
        ),
        0,
        1,
    )

    singletonDKO.nGuidePairs = len(singletonDKO.efficiencies.flatten())
    singletonDKO.nInitialCells = np.random.poisson(
        singletonDKO.nInitialCellsV, size=singletonDKO.nGuidePairs
    )

    if returnCounts:
        (
            final_counts_s,
            sgRNAEssentialities_s,
            init_counts_s,
            lfcs_s,
        ) = singletonDKO.getLFCs(singletons=True, returnCounts=returnCounts)

    else:
        lfcs_s, sgRNAEssentialities_s = singletonDKO.getLFCs(
            singletons=True, returnCounts=returnCounts
        )

    geneIdx = singletonDKO.rnaIdx.reshape(-1, 2)

    # LFC gene indices

    gene1 = singletonDKO.rnaIdx[:, :, 0].T.ravel()
    gene2 = singletonDKO.rnaIdx[:, :, 1].T.ravel()

    # pprint(list(zip(gene1, gene2)))

    # LFC RNA indices

    rnasLib = np.array(range(0, dko.nGenes * dko.nGuidesPerGene))
    rnasSgl = np.array(
        range(dko.nGenes * dko.nGuidesPerGene, dko.nGenes * dko.nGuidesPerGene + 6)
    )  # 3 NE * 2 sgrna

    rna1 = np.repeat(
        rnasLib,
        np.unique(gene1, return_counts=True)[1][0] // singletonDKO.nGuidesPerGene,
    )
    rna2 = np.tile(
        rnasSgl,
        np.unique(gene2, return_counts=True)[1][0] // singletonDKO.nGuidesPerGene,
    )

    if returnCounts:
        return singletonDKO, (
            lfcs_s,
            init_counts_s,
            final_counts_s,
            sgRNAEssentialities_s,
            rna1,
            gene1,
            rna2,
            gene2,
        )
    else:
        return singletonDKO, (lfcs_s, sgRNAEssentialities_s, rna1, gene1, rna2, gene2)


class DoubleKO(object):
    def __init__(
        self,
        nInitialCells=3000,
        nGenes=100,
        nGuidesPerGene=2,
        splitEfficiencies=True,
        seed=42,
        context=None,
        gi_contexts=None,
        gi_context_lists=None,
        geneEssentiality=None,
        synergies=None,
        sgRNAEfficiencies=None,
        pairEfficiency=None,
    ):
        np.random.seed(seed)

        self.nInitialCellsV = nInitialCells
        self.nGenes = nGenes
        self.nGuidesPerGene = nGuidesPerGene

        # Efficiencies multiplied by each term, like in GEMINI, rather than
        # an overall efficiency factor
        self.splitEfficiencies = splitEfficiencies

        # Start with everything at the gene level

        if synergies is None:
            self.synergies = np.random.normal(0.0, 0.01, (self.nGenes, self.nGenes))
            np.fill_diagonal(self.synergies, 0)

            # Symmetrise
            self.synergies = np.tril(self.synergies) + np.tril(self.synergies, -1).T

        else:
            self.synergies = synergies

        # sns.heatmap(self.synergies, cmap=sns.color_palette("vlag", as_cmap=True), vmin = -0.5, vmax = 0.5)
        # plt.savefig('syn_before.pdf')
        # plt.clf()

        if not (gi_contexts is None):
            for i in gi_context_lists:
                self.synergies += gi_contexts[i]

        # sns.heatmap(self.synergies, cmap=sns.color_palette("vlag", as_cmap=True), vmin = -0.5, vmax = 0.5)
        # plt.savefig('syn_after.pdf')
        # plt.clf()

        if geneEssentiality is None:
            # We could even populate this with real data from the essentiality scores
            self.geneEssentiality = np.random.normal(0.05, 0.1, size=self.nGenes)

        else:
            self.geneEssentiality = geneEssentiality

        if not context is None:
            self.context = context
            self.geneEssentiality += self.context

        self.essentialities = self.combinedEssentiality(
            self.geneEssentiality, self.synergies
        )

        if sgRNAEfficiencies is None:
            self.sgRNAEfficiencies = np.clip(
                np.random.normal(0.95, 0.02, size=self.nGenes * self.nGuidesPerGene),
                0,
                1,
            )

        else:
            self.sgRNAEfficiencies = sgRNAEfficiencies

        # From the ACE paper
        self.effComboFactor = np.ones(
            (len(self.sgRNAEfficiencies), len(self.sgRNAEfficiencies))
        )

        if pairEfficiency is None:
            # Efficiency for guide pair term (synergy prefactor)
            self.pairEfficiency = np.clip(
                np.random.normal(
                    0.95,
                    0.02,
                    size=(
                        self.nGenes * self.nGuidesPerGene,
                        self.nGenes * self.nGuidesPerGene,
                    ),
                ),
                0,
                1,
            )

        else:
            self.pairEfficiency = pairEfficiency

        self.pairEfficiency = (
            np.tril(self.pairEfficiency) + np.tril(self.pairEfficiency, -1).T
        )

        self.efficiencies = self.combinedEfficiencies(
            self.sgRNAEfficiencies, self.effComboFactor
        )

        # Effectively the number of plasmids
        self.nGuidePairs = len(self.efficiencies.flatten())

        self.nInitialCells = np.random.poisson(
            self.nInitialCellsV, size=self.nGuidePairs
        )

    def combinedEssEff(
        self,
        geneEssentiality,
        sgRNAEfficiencies,
        pairEfficiency,
        synergies,
        singletons=False,
    ):
        essEffSingle = geneEssentiality[self.rnaIdxToGeneIdx] * sgRNAEfficiencies

        essEffOuter = (
            np.add.outer(essEffSingle, essEffSingle)
            if not singletons
            else np.add.outer(essEffSingle, np.zeros(3 * self.nGuidesPerGene))
        )
        pairTerm = (
            pairEfficiency * synergies[self.rnaIdx[:, :, 0], self.rnaIdx[:, :, 1]].T
        )

        return essEffOuter + pairTerm

    # Something like: (1 - eff_1 - eff_2 - eff_12) * g0 + (eff_1 - eff_12) * g1 + (eff_2 - eff_12) * g2 + eff_12 * g2

    def combinedEssentiality(self, essentiality, synergies=None, singletons=False):
        essentialityMatrix = np.add.outer(
            essentiality,
            essentiality if not singletons else np.zeros(3),
        )

        # Try an absolute value for synergy, so that the result is
        # essentiality gene 1 + essentiality gene 2 + synergy
        # where synergy is in the range[-infinity, infinity]

        if singletons:
            synergies = np.zeros_like(essentialityMatrix)

        return np.clip(essentialityMatrix + synergies, None, 1)

    def combinedEfficiencies(self, efficiency, effComboFactor=None, singletons=False):
        efficiencyMatrix = np.outer(
            efficiency,
            efficiency if not singletons else np.ones(3 * self.nGuidesPerGene),
        )

        # Include a factor that either improves or reduces sgRNA efficiency
        # on a per combination basis, if required

        if singletons:
            effComboFactor = np.ones_like(efficiencyMatrix)

        return np.clip(efficiencyMatrix * effComboFactor, 0, 1)

    def getLFCs(self, poisson=False, singletons=False, returnCounts=False):
        # Map vectors of length nGenes * nGuidesPerGene to the vector of length nGenes
        self.rnaIdxToGeneIdx = np.repeat(
            range(self.nGenes), self.nGuidesPerGene
        )  # [0, 0, 1, 1, 2, 2, ...]

        # Map matrices of size (nGenes, nGenes)
        # to matrices of shape (nGenes * nGuidesPerGene, nGenes * nGuidesPerGene)
        self.rnaIdx = np.dstack(np.meshgrid(self.rnaIdxToGeneIdx, self.rnaIdxToGeneIdx))
        # print(rnaIdx[:,:,0], rnaIdx[:,:,1])
        # => genes[rnaIdx[:,:,0], rnaIdx[:,:,1]].T

        if singletons:
            self.rnaIdxToNEGeneIdx = np.repeat(
                range(3), self.nGuidesPerGene  # 3 non-essentials
            )  # [0, 0, 1, 1, 2, 2, ...]

            self.rnaIdx = np.dstack(
                np.meshgrid(self.rnaIdxToGeneIdx, self.rnaIdxToNEGeneIdx)
            )

        # '...scaling parameter that captures the relationship between sequencing depth and
        # the number of cells infected'
        gamma = np.ones(self.nGuidePairs)

        # '...captures the relationship between sequencing depth and the number of cells
        # that remain after cell growth, but also can accommodate sample specific differences
        # in the growth rate of all cells (independent of sgRNA)'
        gamma_prime = np.ones(self.nGuidePairs)

        # Initial read counts
        x_sg = (
            negativeBinomial(gamma * self.nInitialCells)
            if not poisson
            else np.random.poisson(gamma * self.nInitialCells)
        )

        # Index the gene essentiality for the multiple guides
        sgRNAEssentialities = self.essentialities[
            self.rnaIdx[:, :, 0], self.rnaIdx[:, :, 1]
        ].T

        # d_sg = self.nInitialCells * (1.0 - self.efficiencies * sgRNAEssentialities)

        d_sg = self.nInitialCells * (
            1.0
            - self.combinedEssEff(
                self.geneEssentiality,
                self.sgRNAEfficiencies,
                self.pairEfficiency,
                self.synergies,
                singletons,
            ).ravel()
            if self.splitEfficiencies
            else self.efficiencies * sgRNAEssentialities
        )
        d_sg = np.clip(d_sg, 0, None)  # Sometimes numpy complains if lambda ~ 0

        # Final read counts
        y_sg = (
            negativeBinomial(gamma_prime * d_sg.ravel())
            if not poisson
            else np.random.poisson(gamma_prime * d_sg.ravel())
        )

        y_sg[y_sg == 0] = 1

        lfcs = np.log2((y_sg / self.nInitialCells) + 1e-8)

        if not returnCounts:
            return lfcs, sgRNAEssentialities
        else:
            return y_sg, sgRNAEssentialities, self.nInitialCells, lfcs

    def calibrationLFCs(
        self,
        offsets,
        n=100,
        negControlLFCWidth=0.05,
        posControlLFCWidth=0.1,
        negControlEssWidth=0.05,
        posControlEssWidth=0.05,
        returnCounts=False,
    ):
        # Forget about gene/guides for now
        # Keep offset the same for pos, neg controls (just a shift)
        # Pos control essentiality is 0.7

        lfcs = []
        for offset in offsets:
            # negControlEss = 0.0
            # posControlEss = 0.8

            # One guide per gene here
            nControlInitialCells = np.random.poisson(self.nInitialCellsV, n)

            negControlEss = np.random.normal(0.0, negControlEssWidth, size=n)
            posControlEss = np.clip(
                np.random.normal(0.7, posControlEssWidth, size=n), None, 1.0
            )

            y_neg = negativeBinomial((1 - negControlEss) * nControlInitialCells)
            y_pos = negativeBinomial((1 - posControlEss) * nControlInitialCells)

            y_neg[y_neg == 0] = 1
            y_pos[y_pos == 0] = 1

            lfcs_neg = np.log2((y_neg / nControlInitialCells) + 1e-8)
            lfcs_neg += np.random.normal(offset, negControlLFCWidth)

            lfcs_pos = np.log2((y_pos / nControlInitialCells) + 1e-8)
            lfcs_pos += np.random.normal(offset, posControlLFCWidth)

            if not returnCounts:
                lfcs.append([lfcs_neg, lfcs_pos])
            else:
                lfcs.append(
                    [
                        nControlInitialCells,
                        np.power(2, lfcs_neg) * nControlInitialCells,
                        np.power(2, lfcs_pos) * nControlInitialCells,
                    ]
                )

        return lfcs

    def plot(self, lfcs, sgRNAEssentialities, init_counts, final_counts):
        sns.histplot(self.efficiencies.ravel(), kde=True)
        plt.savefig("efficiencies.pdf")
        plt.clf()

        sns.histplot(sgRNAEssentialities.ravel(), kde=True)
        plt.savefig("essentialities.pdf")
        plt.clf()

        # DANGER ZONE
        # Remove the long tail, just for plotting
        plotLfcs = lfcs[lfcs > -3]
        plotSGRNAEssentialities = sgRNAEssentialities.ravel()[lfcs > -3]

        # TODO: average over guides, add replicates, etc...

        sns.histplot(plotLfcs, kde=True)
        plt.xlabel("value")
        plt.savefig("ace_DKO_lfcs.pdf")
        plt.clf()

        plt.plot(plotLfcs, plotSGRNAEssentialities, ".", markersize=1.0, alpha=0.25)
        plt.xlabel("value")
        plt.ylabel("Combined essentiality")
        plt.savefig("lfc_ess.pdf")
        plt.clf()

        highEss = plotLfcs[
            (plotSGRNAEssentialities > np.quantile(plotSGRNAEssentialities, 0.8))
            & (plotSGRNAEssentialities < 1.0)
        ]
        lowEss = plotLfcs[
            (plotSGRNAEssentialities < np.quantile(plotSGRNAEssentialities, 0.2))
        ]

        sns.kdeplot(highEss, label="High essentiality")
        sns.kdeplot(lowEss, label="Low essentiality")
        plt.xlabel("value")
        plt.legend(loc=0)
        plt.savefig("lfcSlice.pdf")
        plt.clf()

        highEss -= np.mean(highEss)
        lowEss -= np.mean(lowEss)

        sns.kdeplot(highEss, label="High essentiality (mean subtracted)")
        sns.kdeplot(lowEss, label="Low essentiality (mean subtracted)")
        plt.xlabel("value")
        plt.legend(loc=0)
        plt.savefig("lfcSliceShifted.pdf")
        plt.clf()


def makeDataset(outDir):
    returnCounts = True
    # nCellLines = 10
    # nGenes = 100
    # nCellLines = 22
    # nGenes = 444
    nCellLines = 5
    nGenes = 100
    nContexts = 5
    nVariantFrac = 0.10

    t = time.time()

    # ms = getGIContextMatrix(num_genes = nGenes, num_contexts = nContexts, total_cell_lines = nCellLines, fraction_gene_pairs_in_context = 0.01)
    #
    # print(ms[0])
    # print(len(ms))
    # print(ms[0].shape)
    #
    # sns.heatmap(ms[0], cmap=sns.color_palette("vlag", as_cmap=True))
    # plt.ylabel("Gene")
    # plt.xlabel("Gene")
    # plt.savefig("contexts0.pdf")
    # plt.clf()
    #
    # sns.heatmap(ms[1], cmap=sns.color_palette("vlag", as_cmap=True))
    # plt.ylabel("Gene")
    # plt.xlabel("Gene")
    # plt.savefig("contexts1.pdf")
    # plt.clf()

    context_matrices = generate_context_matrices(nGenes, nContexts, 0.01, scale = 0.1)
    cell_line_to_contexts = assign_contexts_to_cell_lines(nCellLines, nContexts, unique_contexts = True)

    sns.heatmap(context_matrices[0], cmap=sns.color_palette("vlag", as_cmap=True))
    plt.ylabel("Gene")
    plt.xlabel("Gene")
    plt.savefig("contexts0.pdf")
    plt.clf()

    sns.heatmap(context_matrices[1], cmap=sns.color_palette("vlag", as_cmap=True))
    plt.ylabel("Gene")
    plt.xlabel("Gene")
    plt.savefig("contexts1.pdf")
    plt.clf()

    pickle.dump((cell_line_to_contexts, context_matrices), open('gi_contexts.pkl', 'wb'))

    contexts = getContextMatrix(
        nCellLines, nGenes, nContexts, nVariantFrac, variance=0.1
    )

    contextsPlot = contexts.copy()
    contextsPlot[contextsPlot == 1.0] = np.nan

    # Maybe I should just always return counts (init, final) and just not use them
    # otherwise?

    # Make a symmetric range about 0
    maxVal = np.max(np.abs(contextsPlot.ravel()))

    # sns.heatmap(contextsPlot, cmap=sns.color_palette("vlag", as_cmap=True), vmin = -maxVal, vmax = maxVal)
    # plt.ylabel("Cell line")
    # plt.xlabel("Gene")
    # plt.savefig("contexts.pdf")
    # plt.clf()

    sgRNAEfficiencies1 = np.clip(
        # np.random.normal(0.65, 0.02, size=nGenes * 1),
        np.random.normal(0.95, 0.02, size=nGenes * 1),
        0,
        1,
    )

    sgRNAEfficiencies2 = np.clip(
        np.random.normal(0.95, 0.02, size=nGenes * 1),
        0,
        1,
    )

    sgRNAEfficiencies = np.concatenate((sgRNAEfficiencies1, sgRNAEfficiencies2))

    sgRNAEfficiencies = np.vstack((sgRNAEfficiencies1, sgRNAEfficiencies2)).reshape(
        (-1,), order="F"
    )

    dko = DoubleKO(
        nGenes=nGenes,
        context=contexts[0],
        gi_contexts=context_matrices,
        gi_context_lists=cell_line_to_contexts[0],
        sgRNAEfficiencies=sgRNAEfficiencies,
    )

    # sns.heatmap(dko.synergies[:10, :10], cmap=sns.color_palette("vlag", as_cmap=True), vmin = -0.17, vmax = 0.17)
    # plt.xlabel('Gene 1')
    # plt.ylabel('Gene 2')
    # plt.savefig('syn_mat.png', dpi = 300)
    # exit(0)

    final_counts, sgRNAEssentialities, init_counts, lfcs = dko.getLFCs(
        returnCounts=returnCounts
    )

    dko.plot(
        lfcs, sgRNAEssentialities, init_counts=init_counts, final_counts=final_counts
    )

    dfCombs = populateCombinationDF(dko, returnCounts=returnCounts)

    print("adding cell lines", time.time() - t)
    dfCombs, dkos = addCellLines(
        dko,
        dfCombs,
        nCellLines,
        contexts=contexts,
        gi_contexts=context_matrices,
        gi_context_lists=cell_line_to_contexts,
        # offsets=offsets,
        returnCounts=returnCounts,
    )

    # if not returnCounts:
    print("adding combo replicates", time.time() - t)
    dfCombs = addReplicates(dfCombs, 3, returnCounts=returnCounts)

    # So the replicates can be projected out
    # Sloooow
    dfCombs = dfCombs.sort_values(
        ["guide_pair_index", "guide1_index", "cell_line"]
    ).reset_index()

    # The same by our new definition
    dfCombs["gene_unq_pair_index"] = dfCombs["gene_pair_index"]
    dfCombs["cell_line_index"] = dfCombs["cell_line"]
    dfCombs["gene1_unq_index"] = dfCombs["g1_idx"]
    dfCombs["gene2_unq_index"] = dfCombs["g2_idx"]

    dfCombs.to_parquet(f"{outDir}/dfCombs_ace.pq")

    print("making singletons", time.time() - t)
    dfSgl = makeSingletonsDF(
        dkos,
        # offsets=offsets,
        returnCounts=returnCounts,
    )

    # Offset so they don't clash with the combs
    dfSgl["guide_pair_index"] = (
        dfSgl["guide_pair_index"] + np.max(dfCombs["guide_pair_index"]) + 1
    )

    # if not returnCounts:
    print("adding singleton replicates", time.time() - t)
    dfSgl = addReplicates(dfSgl, 3, returnCounts=returnCounts)

    dfSgl["cell_line_index"] = dfCombs["cell_line"]
    # The same by our new definition
    dfSgl["gene_unq_pair_index"] = dfSgl["gene_pair_index"]
    dfSgl["gene1_unq_index"] = dfSgl["g1_idx"]
    dfSgl["guide1_index"] = dfSgl["guide1_index_s"]

    dfSgl = dfSgl.sort_values(
        ["guide1_index_s", "guide2_index_s", "cell_line"]
    ).reset_index()
    dfSgl.to_parquet(f"{outDir}/dfSgl_ace.pq")

    # Offsets to test calibration

    nCalib = 100  # Number of genes for calibration

    # The true offset, not one approximated by essentiality close to zero
    offsets = np.concatenate(
        (np.zeros(1), np.random.uniform(-0.5, 0.5, nCellLines - 1))
    )

    calibData = np.array(
        dko.calibrationLFCs(
            offsets, negControlLFCWidth=0.01, n=nCalib, returnCounts=returnCounts
        )
    )

    plt.hist(calibData.ravel(), bins=100)
    plt.savefig("calib.pdf")
    plt.clf()

    # Offset according to singletons (and therefore also combinations)
    guide_pair_index_c = (
        np.array(len(calibData[:, 0, :].ravel()))
        + np.max(dfCombs["guide_pair_index"])
        + 1
    )

    dfCalib = pd.DataFrame(
        {
            "plasmid": calibData[:, 0, :].ravel(),
            "value": calibData[:, 1, :].ravel(),
            "value_pos": calibData[:, 2, :].ravel(),
            "guide_pair_index": guide_pair_index_c,
            "cell_line": np.tile(range(nCellLines), [nCalib, 1]).T.ravel(),
        }
    )

    dfCalib["cell_line_index"] = dfCalib["cell_line"]

    dfCalib.to_parquet(f"{outDir}/dfCalib_ace.pq")

    offsetsCounts = np.zeros(len(offsets))
    for i, dko in enumerate(dkos):
        offsetsCounts[i] = (
            np.power(2, offsets[i]) * dko.nInitialCellsV - dko.nInitialCellsV
        )

    # Save as LFC not as counts - recalculate these!
    dfOffsets = pd.DataFrame(
        {
            "offsets": offsets.ravel(),
            "offsetsCounts": offsetsCounts.ravel(),
        }
    )
    dfOffsets.to_parquet(f"{outDir}/dfOffsets_ace.pq")


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--out-dir",
        type=str,
        dest="out_dir",
        default=".",
        help="Output directory for the simulated data.",
    )

    args = argParser.parse_args()

    makeDataset(outDir=args.out_dir)
