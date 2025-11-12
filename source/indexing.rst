Data indices
------------

Indices in the dataset are Valinor's way of knowing what observations correspond to the same guides, genes, and experiments. Observations with the same `gene_1_index` have the same gene in the first position, and `gene_2_index` observations have the same gene in the second position. Similar is true for `guide` indices, and those observations with the same `cell_line_index` are from the same experiment. This allows Valinor to share parameters across experiments and genes, making the outputs more robust.

The index (and other) variables required for Valinor can be found in :doc:`required_variables`. These can either be calculated manually, or constructed using the `data_preparation/prepare_data.py` script. This script takes CSV, HDF5, or Parquet files as input along with the names of the columns that correspond to the initial counts, final counts, gene names, guide names, and experiment/cell line names. This will output Parquet files corresponding to the combination data, and controls/singletons data if it exists. ::

    python data_preparation/prepare_data.py
        --input_file input.csv
        --initial_counts_column initial_counts
        --final_counts_column final_counts
        --gene_1_column gene_1
        --gene_2_column gene_2
        --guide_1_column guide_1
        --guide_2_column guide_2
        --experiment_column cell_line
        --output_file output

Replicates will be handled transparently, as it is assumed that these are observations that have otherwise identical parameters (gene names, experiment, etc).
