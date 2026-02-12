
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Read data
data = pd.read_csv("/Users/karime/Downloads/datasets.csv")
print("Preview:")
print(data.head())

# Assumptions on column names
# - 'dataset' column identifies dataset name/category
# - 'x' and 'y' columns are coordinates
required_cols = ["dataset", "x", "y"]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found columns: {list(data.columns)}")

# 1) Number of datasets
unique_datasets = data["dataset"].dropna().unique()
print("\nNumber of datasets:", len(unique_datasets))

# 2) Names of datasets
print("Dataset names:", list(unique_datasets))

# 3) Statistics per dataset (count, mean, variance, std dev) for x and y
stats = (
    data.groupby("dataset")[['x', 'y']]
        .agg(count=('x', 'count'),
             x_mean=('x', 'mean'), x_var=('x', 'var'), x_std=('x', 'std'),
             y_mean=('y', 'mean'), y_var=('y', 'var'), y_std=('y', 'std'))
)
print("\nPer-dataset statistics:")
print(stats)

# 4) Observations (basic): spread and central tendency quick summary
print("\nObservations:")
# Example observations: datasets with largest variance in x and y
max_x_var_ds = stats['x_var'].idxmax()
max_y_var_ds = stats['y_var'].idxmax()
min_x_var_ds = stats['x_var'].idxmin()
min_y_var_ds = stats['y_var'].idxmin()
print(f"- Largest x variance: {max_x_var_ds} ({stats.loc[max_x_var_ds, 'x_var']:.3f})")
print(f"- Largest y variance: {max_y_var_ds} ({stats.loc[max_y_var_ds, 'y_var']:.3f})")
print(f"- Smallest x variance: {min_x_var_ds} ({stats.loc[min_x_var_ds, 'x_var']:.3f})")
print(f"- Smallest y variance: {min_y_var_ds} ({stats.loc[min_y_var_ds, 'y_var']:.3f})")

# 5) Violin plots of x-coordinates per dataset
plt.figure(figsize=(1.5*len(unique_datasets) + 4, 5))
sns.violinplot(data=data, x='dataset', y='x', inner='quartile', cut=0)
plt.title('Violin plot of x by dataset')
plt.xlabel('Dataset')
plt.ylabel('x')
plt.tight_layout()
plt.savefig('/Users/karime/Documents/VSC/ML/figures/violin_x.png', dpi=300, bbox_inches='tight')
plt.close()

# 6) Violin plots of y-coordinates per dataset
plt.figure(figsize=(1.5*len(unique_datasets) + 4, 5))
sns.violinplot(data=data, x='dataset', y='y', inner='quartile', cut=0)
plt.title('Violin plot of y by dataset')
plt.xlabel('Dataset')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('/Users/karime/Documents/VSC/ML/figures/violin_y.png', dpi=300, bbox_inches='tight')
plt.close()

# 7) Correlation between x and y for each dataset
corrs = data.groupby('dataset').apply(lambda g: g['x'].corr(g['y']))
print("\nCorrelation between x and y per dataset:")
print(corrs)

# 8) Covariance matrix for each dataset
print("\nCovariance matrix for each dataset:")
for dataset_name in unique_datasets:
    dataset_group = data[data['dataset'] == dataset_name][['x', 'y']]
    cov_matrix = dataset_group.cov()
    print(f"\n{dataset_name}:")
    print(cov_matrix)

# 9) Linear regression between x and y for each dataset
print("\nLinear regression results (slope, intercept, r-value) for each dataset:")
regression_results = {}
for dataset_name in unique_datasets:
    dataset_group = data[data['dataset'] == dataset_name]
    x_vals = dataset_group['x'].values
    y_vals = dataset_group['y'].values
    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    regression_results[dataset_name] = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value
    }
    print(f"\n{dataset_name}:")
    print(f"  Slope: {slope:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    print(f"  R-value: {r_value:.6f}")

# 10) Scatterplots for all datasets using FacetGrid
g = sns.FacetGrid(data, col='dataset', col_wrap=3, height=4, aspect=1)
g.map_dataframe(sns.scatterplot, x='x', y='y')
g.set_titles("{col_name}")
plt.tight_layout()
plt.savefig('/Users/karime/Documents/VSC/ML/figures/scatterplots_facetgrid.png', dpi=300, bbox_inches='tight')
plt.close()

# 11) Scatterplots with regression lines for all datasets using lmplot
for i, dataset_name in enumerate(unique_datasets):
    dataset_group = data[data['dataset'] == dataset_name]
    sns.lmplot(data=dataset_group, x='x', y='y', height=5, aspect=1)
    plt.suptitle(f'Scatterplot with regression line: {dataset_name}', y=1.00)
    plt.tight_layout()
    plt.savefig(f'/Users/karime/Documents/VSC/ML/figures/scatterplot_regression_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nAll plots have been saved to the figures directory.")
