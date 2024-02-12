import numpy as np

import pandas as pd

from tqdm import tqdm

import copy


def makeSingletonsDF(dkos, offsets=None, returnCounts=False, ncGenes=3):
    # TO DO: Use a separate make cell lines function, like for combinations

    dfs = []

    for i, dko in tqdm(enumerate(dkos)):
        singletonDKO, values = makeSingletons(
            dko, returnCounts=returnCounts, ncGenes=ncGenes
        )

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


def makeSingletons(dko, returnCounts=False, ncGenes=3):
    singletonDKO = copy.copy(dko)

    singletonDKO.synergies = np.zeros((singletonDKO.nGenes, ncGenes))

    singletonDKO.essentialities = singletonDKO.combinedEssentiality(
        singletonDKO.geneEssentiality,
        singletons=True,
        ncGenes=ncGenes,
    )

    singletonDKO.efficiencies = singletonDKO.combinedEfficiencies(
        singletonDKO.sgRNAEfficiencies,
        singletons=True,
        ncGenes=ncGenes,
    )

    # Efficiency for guide pair term, sample separately for
    # singleton - NE combinations
    singletonDKO.pairEfficiency = np.clip(
        np.random.normal(
            0.95,
            0.02,
            size=(
                singletonDKO.nGenes * singletonDKO.nGuidesPerGene,
                ncGenes * singletonDKO.nGuidesPerGene,
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
        ) = singletonDKO.getLFCs(
            singletons=True, returnCounts=returnCounts, ncGenes=ncGenes
        )

    else:
        lfcs_s, sgRNAEssentialities_s = singletonDKO.getLFCs(
            singletons=True, returnCounts=returnCounts, ncGenes=ncGenes
        )

    geneIdx = singletonDKO.rnaIdx.reshape(-1, 2)

    # LFC gene indices

    gene1 = singletonDKO.rnaIdx[:, :, 0].T.ravel()
    gene2 = singletonDKO.rnaIdx[:, :, 1].T.ravel()

    # LFC RNA indices

    rnasLib = np.array(range(0, dko.nGenes * dko.nGuidesPerGene))
    rnasSgl = np.array(
        range(
            dko.nGenes * dko.nGuidesPerGene,
            dko.nGenes * dko.nGuidesPerGene + ncGenes * dko.nGuidesPerGene,
        )
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
