#%%
import pandas as pd
from DataTool import DataLoader
data_loader = DataLoader()

def evaluate_base_line(baseline_csv_path, saved_csv_path):
     print('Evaluating baseline...')
     baseline = data_loader.load_csv_to_pandas(baseline_csv_path)
     
     sorted_baseline = baseline.sort_values(by='id')
     
     grouped_results = sorted_baseline.groupby(['resampler']) \
                                     .agg(recall_mean=('recall', 'mean'),
                                          recall_std=('recall', 'std'),
                                          auc_mean=('auc', 'mean'),
                                          auc_std=('auc', 'std'),
                                          gmean_mean=('gmean', 'mean'),
                                          gmean_std=('gmean', 'std'),
                                          precision_mean=('precision', 'mean'),
                                          precision_std=('precision', 'std'),
                                          fscore_mean=('fscore', 'mean'),
                                          fscore_std=('fscore', 'std'))
                                          
     
     
     #save to csv
     grouped_results.to_csv(saved_csv_path)
     print('Finish evaluating baseline...')

if __name__ == '__main__':
    baseline_csv_path = './result/german_new_base/result.csv'
    saved_csv_path    = './result/baseline/german-0406.csv'
    evaluate_base_line(baseline_csv_path, saved_csv_path)
