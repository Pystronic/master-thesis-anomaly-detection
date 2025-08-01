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

for (category, metric) in itertools.product(categories, metric_columns):
    filtered = frame.loc[frame['category'] == category]
    metric_values = np.array([list(filtered.loc[filtered['model'] == model][metric]) for model in models])

    # ToDo: min, max, variance, mean
    # TodO: plot per model

# ToDo:
# Pro categorie + metrik, plot zum vergleich der Werte der Modelle
# Max + Min Wert & Model,
# Varianz + Mean berechnen & im Plot ausgeben