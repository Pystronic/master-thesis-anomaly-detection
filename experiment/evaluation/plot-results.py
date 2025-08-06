import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

RESULT_FILE = 'test_results.csv'

# Read file
frame = pd.read_csv(RESULT_FILE, delimiter=',')

# For each category
# Boxplot each value
metric_columns = [x for x in frame.columns if (x != 'category') & (x != 'model') & ('Unnamed' not in x)]
categories = frame['category'].unique() # ToDo: maybe order
models = frame['model'].unique() # ToDo: sort

print('#### Categories ####')
print(categories)
print('#### Models ####')
print(models)
print('#### Metrics ####')
print(metric_columns)

# Boxplot for each metric per categorie
for category in categories:
    filtered = frame.loc[frame['category'] == category]
    metric_values = [list(filtered[metric]) for metric in metric_columns]

    # ToDo: Print / save images
    # ToDo: check style
    # ToDo: Rotate labels
    plt.boxplot(metric_values, orientation='vertical', tick_labels=metric_columns)
    plt.title(f'Boxplots für {category}')
    plt.show()

# Analyse each metric per category
categories = np.array([])
metrics = np.array([])
mins = np.array([])
min_models = np.array([])
maxes = np.array([])
max_models = np.array([])
variances = np.array([])
means = np.array([])

for (category, metric) in itertools.product(categories, metric_columns):
    filtered = frame.loc[frame['category'] == category]
    metric_values = np.array([list(filtered.loc[filtered['model'] == model][metric]) for model in models])

    min = metric_values.min()
    max = metric_values.max()
    min_model = models[metric_values.argwhere(lambda x: x == min)]
    max_model = models[metric_values.argwhere(lambda x: x == max)]

    categories.append(category)
    metrics.append(metric)
    mins.append(min)
    min_models.append(min_model)
    maxes.append(max)
    max_models.append(max_model)
    variances.append(metric_values.var())
    means.append(metric_values.mean())

    plt.bar(metric_values, tick_label=models)
    plt.title(f'Werte pro Model für {metric} in {category}')
    plt.show()

frame = pd.DataFrame({
    'Category': categories,
    'Metric': metrics,
    'Min': mins,
    'Min Model': min_models,
    'Max': maxes,
    'Max Models': max_models,
    'Variance': variances,
    'Mean': means
})
frame.to_csv('./metric_evaluation.csv')

# Pro categorie + metrik, plot zum vergleich der Werte der Modelle