Data indices
------------

Valinor uses NumPy style 'fancy' indexing (`advanced indexing <https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing>`_) so that expressions can be evaluated as vector calculations for each element in the dataset at once, whilst also ensuring that samples from the parameter distributions are re-used as appropriate. Put another way, this indexing maps, one-to-many, the probability distributions of the internal parameters to the observations.

For example, to evaluate the predicted counts of a single knock-out experiment that consists of four data points, corresponding to two replicates of two different genes knocked out, we would have an array of length 2 that describes the effect of the knockout of each gene,

::

    g = [g_0, g_1]

and these would represent the shared parameters for each of the two replicates for that genetic knock out. The initial and final counts for our experiment are arrays of length 4 each that are different for each

::

    N = [N_0, N_1, N_2, N_3]

and similarly for N_initial. N_0 and N_1 are replicates of each other, and so are N_2 and N_3 and therefore will share g parameters (g_0 and g_1, respectively). Written out in full, we have an equation that describes the expectation value of each these counts according to our internal parameters, g,

::

    N[0] = N_initial[0] * exp(g[0])
    N[1] = N_initial[1] * exp(g[0])
    N[2] = N_initial[2] * exp(g[1])
    N[3] = N_initial[3] * exp(g[1])

For clarity and to benefit from vector operations on CPUs and GPUs, it would be nice to express this as a single vector equation that looks more like

::

    N = N_initial * exp(g)

however as N is length 4 and g is length 2, it's not clear what operation should be performed here. However using NumPy style fancy indexing, we can define a map, ``gene_to_data`` such that

::

    g[gene_to_data] = [g[0], g[0], g[1], g[1]]

so that this is the same shape as our `N` array. This ``gene_to_data`` map is a length `N` array that corresponds to the correct index in the gene term array for that data index, so here this is

::

    gene_to_data = [0, 0, 1, 1]

The convention in Valinor is that these index arrays should always map *from* parameters *to* the data array (not the other way around), so should **always** be the same length as the total number of data points (here, ``len(N)``). These should also **always** correspond to indices that are in the parameter array, so here their maximum must be less than ``len(gene_terms) - 1`` (NumPyro will not check this for you!).

Although seemingly a little convoluted, this allows Valinor to be flexible with respect to the measurements that inform each parameter, and enables complex experimental designs to be handled transparently.
