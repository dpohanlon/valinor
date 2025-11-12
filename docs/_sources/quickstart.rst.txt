Quick start
=================

Once Valinor is installed, it can be run using the `valinor` executable: ::

    valinor --combinationsFile combinations.pq
            --singletonsFile singletons.pq
            --epochs 1000 --learningRate 0.01


Where `combinations.pq` and `singletons.pq` are data files that contain sequencing counts, and indices that describe the relations between the observations (see :doc:`indexing`). This configures Valinor to run for 1000 epochs with a learning rate of 0.01, and using a dataset that contains 'combinations' (where two genes of interest are knocked out) and 'singletons' (where only one of the genes present in the combinations in knocked out). If you have a GPU installed (which is highly recommended for larger datasets) this will be automatically configured to use it.

Valinor will then perform the fit, by default, in two stages: first fitting only the single knock-out data, and then fitting the combined double and single knock-out data. This stabilises the model convergence and aids identifiability. The loss over the course of the fit is also plotted in `valinor_loss_combs.pdf`.

Outputs
--------

Valinor will output several files corresponding to the posterior distributions of the parameters of each separate model, as Pandas DataFrames.::

    CombsModel.pq
    FirstStageSingletonsModel.pq
    SecondStageSingletonsModel.pq


In the above case, this will be a file named `CombsModel.pq`, which contains the posterior distribution of the parameters of the model that fits the combined double and single knock-out data, and `SecondStageSingletonsModel.pq`, which contains the posterior distribution of the parameters of the model that fits the single knock-out data only. In the two-stage model, there will also be a file named `FirstStageSingletonsModel.pq`, which contains the result of the first stage fit of the single knock-out data.

In general in these files, columns ending in `_mean` are the mean of the posterior parameter distribution estimated by sampling from the model, and columns ending in `_std` are the standard deviation of this distribution. As Valinor uses Variational inference, these statistics summarise the full information within these distributions.

The columns `ko_growth_s_mean` corresponds to the growth factor of the knock-out genotype, relative to the estimated null growth, for single KO. For the combinations `ko_growth_1_mean`, `ko_growth_2_mean`, correspond to the expectation value of the gene 1 and 2 growth factors, respectively, and `ko_growth_12_mean` corresponds to the scale of the genetic interaction, above and beyond the product of the single knock-out growth factors (additive on a log-scale). A value of 0 indicates no effect relative to the null, whereas a large negative value indicates a strong genetic interaction that acts to kill more cells, and a large positive value indicates a strong genetic interaction that acts to kill fewer cells.

Also output are posterior predictive distributions for the observations, `*obs.h5`, to be compared with the real observed data.::

    obs.h5
    obs_init.h5
    obs_s.h5
    obs_s_init.h5

These can be used to assess the fit of the model to the data, and typically are oversampled by the same number of samples as for the estimates of the mean and standard deviation of the posterior distributions, in order to reduce the variance of the estimates.

Report
------

It is also possible to produce an interactive HTML report that guides you through the various results and diagnostic plots with `valinorreport`. ::

    valinorreport
        --val_combo CombsModel.pq
        --val_single SecondStageSingletonsModel.pq
        --data_combo combinations.pq
        --data_single singletons.pq
        --valinorLossFile valinor_loss.svg
        --valinorConfigFile valinorrun.json
        --output_file valinoroutput_processed.pq
        --report_folder valinor_report

This will produce a directory `valinor_report` that contains HTML files that summarise your data. Get started by double clicking on the `index.html` file. Sometimes the size of the output file can exceed that permitted by your web browser, and so in that case you should subset to the top hits with the command `subsetSLpairs`.

Simulation
----------

A simulation of double knock out screens is also present as `valinor_sim`. This can be used to evaluate the performance of the model and any downstream analysis on the experimental design, and parameters such as the number of guides per gene and the number of replicates. This also includes context-dependent genetic interactions across subsets of cell lines. At the moment the experimenta designs are restricted to all-by-all gene pairs, but these can be post-processed to create more elaborate constructions.

The simulation can be run as follows ::

    valinor_sim --nGenes 30 --nCellLines 10
