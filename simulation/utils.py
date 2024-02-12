import numpy as np
import functools

from tqdm import tqdm

import pandas as pd


def negativeBinomial(mean, variance=None, od=10, size=None):
    # Minimum observable
    mean[mean < 1e-8] = 1

    if variance is None:
        variance = od * mean

    n_nb = -(mean**2 / (mean - variance))
    p_nb = 1.0 - (mean / (variance + 1e-8))

    p_nb = np.clip(p_nb, 0, 1)

    return np.random.negative_binomial(np.maximum(1e-6, n_nb), 1.0 - p_nb, size=size)


def genePairStr_(g1, g2):
    return str(g1) + "~" + str(g2)


@functools.lru_cache(maxsize=None)
def genePairStr(g1, g2):
    if g2 > g1:
        return genePairStr_(g1, g2)
    else:
        return genePairStr_(g2, g1)


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
