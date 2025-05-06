import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

import seaborn as sns

import numbers

import numpy as np

from valinor_simulation.utils import negativeBinomial


class DoubleKO(object):
    def __init__(
        self,
        prototype = True,
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
        od=20,
        od_init=2,
    ):
        np.random.seed(seed)

        self.prototype = prototype

        # Can be either a containers of counts or the central value to
        # generate from

        if isinstance(nInitialCells, numbers.Number):
            self.nInitialCellsV = nInitialCells
        else:
            self.nInitialCells = nInitialCells
        self.nGenes = nGenes
        self.nGuidesPerGene = nGuidesPerGene

        # Efficiencies multiplied by each term, like in GEMINI, rather than
        # an overall efficiency factor
        self.splitEfficiencies = splitEfficiencies

        self.od = od
        self.od_init = od_init

        # Start with everything at the gene level

        if synergies is None:
            # Additive on essentiality, but modified multiplicatively by context
            self.synergies = np.random.normal(0.0, 0.25, (self.nGenes, self.nGenes))
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
                self.synergies *= gi_contexts[i]

        # sns.heatmap(self.synergies, cmap=sns.color_palette("vlag", as_cmap=True), vmin = -0.5, vmax = 0.5)
        # plt.savefig('syn_after.pdf')
        # plt.clf()

        if geneEssentiality is None:
            # We could even populate this with real data from the essentiality scores
            self.geneEssentiality = np.random.normal(0.05, 0.2, size=self.nGenes)

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
        ) if prototype else self.nInitialCells

    def combinedEssEff(
        self,
        geneEssentiality,
        sgRNAEfficiencies,
        pairEfficiency,
        synergies,
        singletons=False,
        ncGenes=3,
    ):
        essEffSingle = geneEssentiality[self.rnaIdxToGeneIdx] * sgRNAEfficiencies

        essEffOuter = (
            np.add.outer(essEffSingle, essEffSingle)
            if not singletons
            else np.add.outer(essEffSingle, np.zeros(ncGenes * self.nGuidesPerGene))
        )
        pairTerm = (
            pairEfficiency * synergies[self.rnaIdx[:, :, 0], self.rnaIdx[:, :, 1]].T
        )

        return essEffOuter + pairTerm

    # Something like: (1 - eff_1 - eff_2 - eff_12) * g0 + (eff_1 - eff_12) * g1 + (eff_2 - eff_12) * g2 + eff_12 * g2

    def combinedEssentiality(
        self, essentiality, synergies=None, singletons=False, ncGenes=3
    ):
        essentialityMatrix = np.add.outer(
            essentiality,
            essentiality if not singletons else np.zeros(ncGenes),
        )

        # Try an absolute value for synergy, so that the result is
        # essentiality gene 1 + essentiality gene 2 + synergy
        # where synergy is in the range[-infinity, infinity]

        if singletons:
            synergies = np.zeros_like(essentialityMatrix)

        return np.clip(essentialityMatrix + synergies, None, 1)

    def combinedEfficiencies(
        self, efficiency, effComboFactor=None, ncGenes=3, singletons=False
    ):
        efficiencyMatrix = np.outer(
            efficiency,
            efficiency if not singletons else np.ones(ncGenes * self.nGuidesPerGene),
        )

        # Include a factor that either improves or reduces sgRNA efficiency
        # on a per combination basis, if required

        if singletons:
            effComboFactor = np.ones_like(efficiencyMatrix)

        return np.clip(efficiencyMatrix * effComboFactor, 0, 1)

    def getLFCs(self, poisson=False, singletons=False, returnCounts=False, ncGenes=3):
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
                range(ncGenes), self.nGuidesPerGene  # 3 non-essentials
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
            negativeBinomial(gamma * self.nInitialCells, od=self.od_init)
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
                ncGenes=ncGenes,
            ).ravel()
            if self.splitEfficiencies
            else self.efficiencies * sgRNAEssentialities
        )
        d_sg = np.clip(d_sg, 0, None)  # Sometimes numpy complains if lambda ~ 0

        # Final read counts
        y_sg = (
            negativeBinomial(gamma_prime * d_sg.ravel(), od=self.od)
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

        if not self.prototype:
            print('Only call this on the prototype')
            return

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

            y_neg = negativeBinomial(
                (1 - negControlEss) * nControlInitialCells, od=self.od
            )
            y_pos = negativeBinomial(
                (1 - posControlEss) * nControlInitialCells, od=self.od
            )

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

    def plot(self, lfcs, sgRNAEssentialities, init_counts, final_counts, name=""):
        sns.histplot(self.efficiencies.ravel(), kde=True)
        plt.savefig(f"efficiencies_{name}.pdf")
        plt.clf()

        sns.histplot(sgRNAEssentialities.ravel(), kde=True)
        plt.savefig(f"essentialities_{name}.pdf")
        plt.clf()

        # DANGER ZONE
        # Remove the long tail, just for plotting
        plotLfcs = lfcs[lfcs > -3]
        plotSGRNAEssentialities = sgRNAEssentialities.ravel()[lfcs > -3]

        # TODO: average over guides, add replicates, etc...

        sns.histplot(plotLfcs, kde=True)
        plt.xlabel("value")
        plt.savefig(f"ace_DKO_lfcs_{name}.pdf")
        plt.clf()

        plt.plot(plotLfcs, plotSGRNAEssentialities, ".", markersize=1.0, alpha=0.25)
        plt.xlabel("value")
        plt.ylabel("Combined essentiality")
        plt.savefig(f"lfc_ess_{name}.pdf")
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
        plt.savefig(f"lfcSlice_{name}.pdf")
        plt.clf()

        highEss -= np.mean(highEss)
        lowEss -= np.mean(lowEss)

        sns.kdeplot(highEss, label="High essentiality (mean subtracted)")
        sns.kdeplot(lowEss, label="Low essentiality (mean subtracted)")
        plt.xlabel("value")
        plt.legend(loc=0)
        plt.savefig(f"lfcSliceShifted_{name}.pdf")
        plt.clf()
