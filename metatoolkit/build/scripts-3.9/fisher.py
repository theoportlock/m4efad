#!/home/theop/venv/bin/python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
'''
import functions as f
import argparse
import pandas as pd
import numpy as np
from itertools import permutations
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection

def fisher(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Fisher's exact test odds ratio and p-value for all pairs of columns in a DataFrame.
    Additionally, compute the mean and standard deviation of counts for each pair of columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame where columns are categories and rows are observations.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex of column pairs, containing odds ratio, p-value, 
        mean count and standard deviation for first and second categories.
    
    Example
    -------
    >>> data = {
    ...     'MAM': [False, True, False, False, True, False],
    ...     'Fem': [False, True, False, True, True, False],
    ...     'Fat': [False, True, True, True, False, True],
    ... }
    >>> df = pd.DataFrame(data)
    >>> fisher(df)
    """
    column_pairs = list(permutations(df.columns, 2))
    results = {
        'odds': [],
        'pval': [],
        'source_true_summary': [],
        'source_false_summary': []
    }

    for source, target in column_pairs:
        contingency_table = pd.crosstab(df[target], df[source], dropna=False)
        valid = contingency_table.shape == (2, 2)
        if valid:
            oddsratio, pvalue = fisher_exact(contingency_table)
            true_count = contingency_table.loc[True, True]
            true_total = contingency_table.loc[:, True].sum()
            true_percentage = round(100 * true_count / true_total, 1)
            source_true_summary = f"{true_count}/{true_total} ({true_percentage}%)"
            false_count = contingency_table.loc[True, False]
            false_total = contingency_table.loc[:, False].sum()
            false_percentage = round(100 * false_count / false_total, 1)
            source_false_summary = f"{false_count}/{false_total} ({false_percentage}%)"
        else:
            oddsratio, pvalue = np.nan, np.nan
            source_true_summary = np.nan
            source_false_summary = np.nan
        results['odds'].append(oddsratio)
        results['pval'].append(pvalue)
        results['source_true_summary'].append(source_true_summary)
        results['source_false_summary'].append(source_false_summary)
    index = pd.MultiIndex.from_tuples(column_pairs, names=['source', 'target'])
    result_df = pd.DataFrame(results, index=index)
    result_df['qval'] = fdrcorrection(result_df.pval)[1]
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Calculate Fisher\'s exact test for all pairs of columns in a DataFrame.')
    parser.add_argument('file', type=str, help='Path to the input file.')
    args = parser.parse_args()

    # load data
    file = args.file
    cats = f.load(file)

    # calculate fisher exact
    out = fisher(cats)

    # save fisher results
    f.save(out, f'{file}Fisher')

if __name__ == '__main__':
    main()
