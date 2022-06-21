import pandas as pd
import numpy as np

from scipy.stats import ks_2samp, chisquare, chi2_contingency, gaussian_kde
from scipy.stats import wasserstein_distance

#
# if we want eclude some columns (ex: target, we can use exc_list)
#
def identify_categorical(df, min_distinct=10, exc_list=[]):
    # if you pass the TARGET it will be excluded from the features list
    cat_columns = []
    
    # remove exclusione list
    col_to_analyze = list(set(df.columns) - set(exc_list))
    
    for col in col_to_analyze:
        # identifichiamo come categoriche tutte le colonne che soddisfano questa condizione !!!
        # la soglia la possiamo cambiare (parm. min_distinct)
        if df[col].dtypes == 'object' or df[col].nunique() < min_distinct:
            cat_columns.append(col)
            
    return cat_columns

# (Wikipedia) Pearson's chi-squared test is used to determine whether there is a statistically significant difference 
# between the expected frequencies and the observed frequencies in one or more categories of a contingency table.

# contingency table contains the occurrencies for each value in reference and current set
# contiene il conteggio delle occorrenze del primo dataset e nel secondo
def compute_contingency_table(ref_col, newset_col):
    # we expect here Pandas df cols (ex: df_ref[col])
    # see https://github.com/Azure/data-model-drift/blob/main/tabular-data/utils.py
    
    index = list(set(ref_col.unique()) | set(newset_col.unique()))

    # this is the best way to handle value missing in one of the two
    value_counts_df = pd.DataFrame(ref_col.value_counts(), index=index)
    value_counts_df.columns = ['reference']
    value_counts_df['new'] = newset_col.value_counts()
    # aggiunge 0 per valori mancanti in uno dei dataset
    value_counts_df.fillna(0, inplace=True)

    result = np.array([[value_counts_df['reference'].values], [value_counts_df['new'].values]])
    
    return result, index

# we're using scipy functions (see import)

# min. number of rows to be considered numerical (not categorical)

# we can specify a list of cols to exclude from drift analysis
def identify_data_drift(df_ref, df_new, p_thr=0.01, do_print=True, exc_list=[]):
    all_cols = df_ref.columns
    
    # p_thr is the threshold used for Null Hyp. tests
    
    # check that the two dataframe have the same columns
    
    if list(df_ref.columns) != list(df_new.columns):
        print("Error: The two DataFrame must have the same columns.")
        print("Closing.")
        
        return None
    
    # identify categorical and numerical columns
    cat_cols = identify_categorical(df_ref, exc_list=exc_list)
    # all the rest excluding target
    num_cols = list(set(all_cols) - set(cat_cols) - set(exc_list))
    
    if do_print:
        print()
        print("*** Report on evidences of Data Drift identified ***")
        print()
    
    # compute only once the describe, to get stats
    # this way is faster!
    set1_describe = df_ref.describe().T
    set2_describe = df_new.describe().T
    
    # enforce exclusion list
    col_to_analyze = sorted(list(set(df_ref.columns) - set(exc_list)))
    
    list_drifts = []
    for col in col_to_analyze:
        # per prendere solo le numeriche
        if col in num_cols:
            # analyze numerical columns using Kolmogoros Smirnov test
            stats, p_value = ks_2samp(df_ref[col].values, df_new[col].values)
            
            type = "continuous"
            
            # save also the stats for the column (see df.describe().T)
            # I have reduced the n. of digits to 2
            # we don't take the count
            stats1 = str(list(np.round(set1_describe.loc[col, :].values[1:], 2))) 
            stats2 = str(list(np.round(set2_describe.loc[col, :].values[1:], 2)))
            stats = stats1 + "," + stats2
            
            # compute the wasserstein distance
            was_distance = wasserstein_distance(df_ref[col].values, df_new[col].values)
            # normalize with mean
            was_distance = round(was_distance/(set1_describe.loc[col, 'mean']), 3)
            
            # compute mean difference/mean1
            mean1 = set1_describe.loc[col, 'mean']
            mean2 = set2_describe.loc[col, 'mean']
            delta_mean_norm = round(abs(mean1 - mean2)/mean1, 3)
            
        # solo le categoriche
        if col in cat_cols:
            # compute table with occurrencies
            c_table, index = compute_contingency_table(df_ref[col], df_new[col])

            stats, p_value, dof, _ = chi2_contingency(c_table)
            
            type = "categorical"
            stats = str(c_table)
            
        # p_value can be interpreted as the probability that the two dataset come from the same distribution
        # if it is too low.. they DON'T
        if p_value < p_thr:
            # detected drift
            p_val_rounded = round(p_value, 5)
            
            if do_print:
                print("Identified drift in column:", col)
                print(f"p_value: {p_val_rounded}")
                print()
            
            drift_info = {"Column": col, 
                          "Type" : type, 
                          "p_value": p_val_rounded,
                          "threshold" : p_thr,
                          "stats" : stats
                         }
            if type == "continuous":
                drift_info["was_distance_norm"] = was_distance
                drift_info["delta_mean_norm"] = delta_mean_norm
                
            list_drifts.append(drift_info)
    
    if (len(list_drifts) == 0):
        if do_print:
            print("No evidence found.")
            print()
        
    return list_drifts
