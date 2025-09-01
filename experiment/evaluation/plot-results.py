import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools

PLOT_METRICS = False
ANALYZE_METRICS = False
ANALYZE_WRONG_SIZE = True

matplotlib.use('qt5cairo')

RESULT_FILE = 'test_results.csv'

# Read file
frame = pd.read_csv(RESULT_FILE, delimiter=',')

# For each category
# Boxplot each value
metric_columns = [x for x in frame.columns if (x != 'category') & (x != 'model') & ('Unnamed' not in x)]
# Remove random nan value in list
categories = sorted(frame['category'].unique())
models = sorted(frame['model'].unique())

print('#### Categories ####')
print(categories)
print('#### Models ####')
print(models)
print('#### Metrics ####')
print(metric_columns)

def boxplot_metrics(category, metrics):
    filtered = frame.loc[frame['category'] == category]
    metric_values = [list(filtered[metric]) for metric in metrics]

    plt.boxplot(metric_values, orientation='vertical', tick_labels=metrics)
    plt.title(f'Boxplots für {category}')
    #plt.xticks(rotation=90)
    plt.show()

def scatter_metrics(category, metrics):
    filtered = frame.loc[frame['category'] == category]
    metric_values = np.array([filtered[metric].to_numpy() for metric in metrics]).flatten()

    x_ticks = np.array([np.repeat(x, len(models)) for x in range(len(metrics))]).flatten()
    plt.scatter(x_ticks, metric_values)
    plt.xticks(np.arange(0, len(metrics)), labels=metrics)
    plt.title(f'Scatterplot für {category}')
    #plt.xticks(rotation=90)
    plt.show()



if PLOT_METRICS:
    img_metrics = [x for x in metric_columns if x.startswith('IMG')]
    px_metrics = [x for x in metric_columns if x.startswith('PX')]
    for category in categories:
        boxplot_metrics(category, img_metrics)
        scatter_metrics(category, img_metrics)

        boxplot_metrics(category, px_metrics)
        scatter_metrics(category, px_metrics)

        scatter_metrics(category, ['rel_images_per_second'])
        boxplot_metrics(category, ['rel_images_per_second'])


# Analyse each metric per category
if ANALYZE_METRICS:
    evaluated_categories = np.array([], dtype=str)
    metrics = np.array([], dtype=str)
    mins = np.array([])
    min_models = np.array([], dtype=str)
    maxes = np.array([])
    max_models = np.array([], dtype=str)
    variances = np.array([])
    means = np.array([])

    for (category, metric) in itertools.product(categories, metric_columns):
        print(f'Category {category} --- Metric {metric}')

        metric_values = frame.loc[frame['category'] == category, metric].to_numpy()

        min_val = metric_values.min()
        max_val = metric_values.max()
        min_model = models[np.argsort(metric_values)[0]]
        max_model = models[np.argsort(metric_values)[-1]]

        evaluated_categories = np.append(evaluated_categories, [category])
        metrics = np.append(metrics, [metric])
        mins = np.append(mins, [min_val])
        min_models = np.append(min_models, [min_model])
        maxes = np.append(maxes, [max_val])
        max_models = np.append(max_models, [max_model])
        variances = np.append(variances, [metric_values.var()])
        means = np.append(means, [metric_values.mean()])

        plt.bar(models, metric_values, bottom=min_val)
        plt.ylim(min_val, max_val)
        plt.title(f'Werte pro Model für {metric} in {category}')
        #plt.show()

    frame = pd.DataFrame({
        'Category': evaluated_categories,
        'Metric': metrics,
        'Min': mins,
        'Min Model': min_models,
        'Max': maxes,
        'Max Models': max_models,
        'Variance': variances,
        'Mean': means
    })
    frame.to_csv('./metric_evaluation.csv', float_format='%.4f')

# Pro categorie + metrik, plot zum vergleich der Werte der Modelle
if ANALYZE_WRONG_SIZE:
    wrong_size = pd.read_csv('test_results_wrong_size.csv', delimiter=',')
    wrong_size_categories = sorted(frame['category'].unique())
    wrong_size_models = sorted(frame['model'].unique())

    data = []
    for (model, category) in itertools.product(wrong_size_models, wrong_size_categories):
        if model not in wrong_size['model'].tolist():
            continue

        if category not in wrong_size['category'].tolist():
            continue


        model_frame = frame.loc[frame['model'] == model]
        model_wrong_size = wrong_size.loc[frame['model'] == model]

        metric_diffs = np.zeros(len(metric_columns))

        for i, metric in enumerate(metric_columns):
            wrong_size_metric_val = model_wrong_size.loc[model_wrong_size['category'] == category, metric]
            if len(wrong_size_metric_val) == 0:
                continue

            frame_metric_val = model_frame.loc[model_frame['category'] == category, metric]
            if len(frame_metric_val) == 0:
                print(frame_metric_val)
                continue

            metric_diffs[i] = (
                frame_metric_val.tolist()[0] -
                wrong_size_metric_val.tolist()[0]
            )

        data.append([model, category, *metric_diffs])

    frame = pd.DataFrame(data, columns=['model', 'category', *metric_columns]);
    frame.to_csv('./wrong_size_evaluation.csv', float_format='%.3f')