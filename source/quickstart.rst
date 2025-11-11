Quick-start guide
=================

Once Valinor is installed, it can be run using the `valinor` executable: ::

    valinor --combinationsFile combinations.pq
            --singletonsFile singletons.pq
            --epochs 1000 --learningRate 0.01


This configures Valinor to run for 1000 epochs with a learning rate of 0.01, and using a dataset that contains 'combinations' (where two genes of interest are knocked out) and 'singletons' (where only one of the genes present in the combinations in knocked out).

Valinor will then perform the fit, by default, in two stages: first fitting only the single knock-out data, and then fitting the combined double and single knock-out data. This stabilises the model convergence and aids identifiability.

Valinor will output several files corresponding to the posterior distributions of the parameters of each separate model, as Pandas DataFrames. In the above case, this will be a file named `CombsModel.pq`, which contains the posterior distribution of the parameters of the model that fits the combined double and single knock-out data, and `SecondStageSingletonsModel.pq`, which contains the posterior distribution of the parameters of the model that fits the single knock-out data only. In the two-stage model, there will also be a file named `FirstStageSingletonsModel.pq`, which contains the result of the first stage fit of the single knock-out data.

In general in these files, columns ending in `_mean` are the mean of the posterior parameter distribution estimated by sampling from the model, and columns ending in `_std` are the standard deviation of this distribution. As Valinor uses Variational inference, these statistics summarise the full information within these distributions.

The columns `ko_growth_s_mean` corresponds to the growth factor of the knock-out genotype, relative to the estimated null growth, for single KO. For the combinations `ko_growth_1_mean`, `ko_growth_2_mean`, correspond to the expectation value of the gene 1 and 2 growth factors, respectively, and `ko_growth_12_mean` corresponds to the scale of the genetic interaction, above and beyond the product of the single knock-out growth factors (additive on a log-scale). A value of 0 indicates no effect relative to the null, whereas a large negative value indicates a strong genetic interaction that acts to kill more cells, and a large positive value indicates a strong genetic interaction that acts to kill fewer cells.

Also output are posterior predictive distributions for the observations, `*obs.h5`, to be compared with the real observed data. These can be used to assess the fit of the model to the data, and typically are oversampled by the same number of samples as for the estimates of the mean and standard deviation of the posterior distributions, in order to reduce the variance of the estimates.
