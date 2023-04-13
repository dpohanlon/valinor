import numpy as np

import pandas as pd

# Average over samples from the posterior to pack into a Pandas DataFrame
def averageOverSamples(samples):

    means = {k, np.mean(s, 0) for k, s in samples}
    stds = {k, np.std(s, 0) for k, s in samples}

    return means, std

def createDataFrame(samples, counts):

    means, stds = averageOverSamples(samples)

    df = pd.DataFrame(np.array(counts).flatten(), columns=["samples"])

    for k in samples.keys():
        df[f'{k}_mean'] = means[k]
        df[f'{k}_stds'] = stds[k]

    return df
    
