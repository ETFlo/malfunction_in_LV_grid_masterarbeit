import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config


def plot_sample(Y, x=None, label = None, title=None, save=False, figname=None):

    fig, ax = plt.subplots()
    marker = '.'
    if config.sample_length <= 96:
        markersize = 2
    elif config.sample_length <= 7*96:
        markersize = 1
    else:
        markersize = 0.5

    if not x:
        if isinstance(Y, np.ndarray):
            x = np.linspace(0, len(Y[0]), len(Y[0]))
            for i in range(len(Y)):
                if label:
                    ax.plot(x, Y[i], marker, label = label[i], markersize=markersize)
                else:
                    ax.plot(x, Y[i], marker, markersize=markersize)
        else:
            x = np.linspace(0, len(Y), len(Y))
            if label:
                ax.plot(x, Y, marker, label=label, markersize=markersize)
            else:
                ax.plot(x, Y, marker, markersize=markersize)
    else:
        if isinstance(Y, np.ndarray):
            for i in range(len(Y)):
                if label:
                    ax.plot(x, Y[i], marker, label=label[i], markersize=markersize)
                else:
                    ax.plot(x, Y[i], marker, markersize=markersize)
        else:
            if label:
                ax.plot(x, Y, marker, label=label, markersize=markersize)
            else:
                ax.plot(x, Y, marker, markersize=markersize)

    fig.show()
    if label:
        plt.legend(loc="best", markerscale=10)
    if title:
        plt.title(title)
    if save:
        plt.savefig(figname + '.png')

    return ax

def plot_2D(y, x=None, labels=None, title=None, x_label=None, y_label=None, save=False, figname=None, style='-'):

    fig, ax = plt.subplots()

    if not x:
        if type(y[0]) == list:
            x = np.linspace(0, len(y[0])+1, len(y[0]))
        else:
            x = np.linspace(0, len(y)+1, len(y))

    if labels:
        if type(y[0]) == list:
            for i in list(range(len(y))):
                ax.plot(x, y[i], style, label=labels[i])
        else:
            ax.plot(x, y, style, label=labels)
    else:
        if type(y[0]) == list:
            for i in list(range(len(y))):
                ax.plot(x, y[i], style)
        else:
            ax.plot(x, y, style)

    fig.show()
    if labels:
        plt.legend(loc="best", markerscale=10)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if save:
        plt.savefig(figname + '.png')
        plt.savefig(figname + '.pdf', dpi=fig.dpi, bbox_inches='tight', format='pdf')

    return ax

def plot_time_sweep():
    path = os.path.join(config.models_folder, learning_config['classifier'])
    file = os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"] + "_training_time_sweep" + "_" + 'result.txt')
    figure_name = os.path.join(path, learning_config["dataset"] + "_training_time_sweep")

    f = open(file, 'r')
    text = f.read()
    F_scores = [float(i.split('\n')[0][:4]) for i in text.split('FScore: ')[1:]]
    Precisions = [float(i.split('\n')[0][:5]) for i in text.split('Precision: ')[1:]]
    Recalls = [float(i.split('\n')[0][:5]) for i in text.split('Recall: ')[1:]]
    plot_2D([F_scores, Precisions, Recalls], x=list(range(1,learning_config['number of epochs']+1)), labels=['F-score', 'Precision', 'Recall'], x_label='Training time', y_label = 'Scores', save=True, figname=figure_name)


def plot_hyp_para_tuning():
    path = os.path.join(config.models_folder, learning_config['classifier'])
    file = os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"] + "_hyper_parameter_analysis_on_" + learning_config["hyperparameter tuning"][0] + "_" + 'result.txt')
    figure_name = os.path.join(path, learning_config["dataset"] + "_hyper_parameter_analysis_on_" + learning_config["hyperparameter tuning"][0])

    f = open(file, 'r')
    text = f.read()
    F_scores = [float(i.split('\n')[0][:4]) for i in text.split('FScore: ')[1:]]
    Precisions = [float(i.split('\n')[0][:5]) for i in text.split('Precision: ')[1:]]
    Recalls = [float(i.split('\n')[0][:5]) for i in text.split('Recall: ')[1:]]
    plot_2D([F_scores, Precisions, Recalls], x=learning_config["hyperparameter tuning"][1], labels=['F-score', 'Precision', 'Recall'], x_label=learning_config["hyperparameter tuning"][0] + ' Values', y_label = 'Scores', save=True, figname=figure_name) #'Number of RNN Attention Blocks Values' #learning_config["hyperparameter tuning"][0] +

def plot_grid_search():
    path = os.path.join(config.models_folder, learning_config['classifier'])
    file = os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"] + "_gridsearch_on_" + learning_config["grid search"][0]
                                + "_" + 'result.txt')
    figure_name = os.path.join(path, learning_config["dataset"] + "_gridsearch_on_" + learning_config["grid search"][0])

    f = open(file, 'r')
    text = f.read()
    F_scores = [float(i.split('\n')[0][:4]) for i in text.split('FScore: ')[1:]]
    Precisions = [float(i.split('\n')[0][:5]) for i in text.split('Precision: ')[1:]]
    Recalls = [float(i.split('\n')[0][:5]) for i in text.split('Recall: ')[1:]]
    plot_2D([F_scores, Precisions, Recalls], x=learning_config["grid search"][1], labels=['F-score', 'Precision', 'Recall'], x_label=learning_config["grid search"][0] + ' Values', y_label = 'Scores', save=True, figname=figure_name) #learning_config["grid search"][0] + ' Values'

def plot_estimate_vs_target_by_load(y, y_pred_nn, y_pred_lr, style='-', phase='phase1', setup='A'):
    path = config.load_estimation_folder

    for load in list(y.columns)[0::2]:
        load_name = load.split('_')[0]
        figure_name = os.path.join(path, f'{phase}_setup_{setup}_estimation_{load_name}')
        y_load_P = list(y[load_name + '_P'].values)
        y_load_Q = list(y[load_name + '_Q'].values)
        y_pred_nn_P = list(y_pred_nn[load_name + '_P'].values)
        y_pred_nn_Q = list(y_pred_nn[load_name + '_Q'].values)
        y_pred_lr_P = list(y_pred_lr[load_name + '_P'].values)
        y_pred_lr_Q = list(y_pred_lr[load_name + '_Q'].values)
        #labels=['Target P', 'Target Q', 'Prediction NN P', 'Prediction NN Q', 'Prediction LR P', 'Prediction LR Q']
        labels_P = ['Target P', 'Estimate NN P', 'Estimate LR P']
        labels_Q = ['Target Q', 'Estimate NN Q', 'Estimate LR Q']

        #plot_2D([y_load_P, y_load_Q,y_pred_nn_P, y_pred_nn_Q,y_pred_lr_P, y_pred_lr_Q], labels=labels, title='Load estimation vs. target values', x_label='Timestep', y_label='Load [kW]', save=True, figname=figure_name)
        plot_2D([y_load_P, y_pred_nn_P, y_pred_lr_P ], labels=labels_P,
                title='Load estimation vs. target values', x_label='Timestep', y_label='Load [kW]', save=True,
                figname=figure_name + '_P', style=style)
        plot_2D([y_load_Q, y_pred_nn_Q, y_pred_lr_Q], labels=labels_Q,
                title='Load estimation vs. target values', x_label='Timestep', y_label='Load [kVA]', save=True,
                figname=figure_name + '_Q', style=style)


def plot_dataset_comparison(scores_per_dataset, dataset_labels, save=False, figname=None):
    """Compare classifier performance metrics across two or more datasets.

    Creates a **grouped bar chart** (mean ± 2σ) and **box plots** for the
    four metrics Accuracy, Precision, Recall and FScore.  All fold values
    from all classifier combos are pooled per dataset.

    Parameters
    ----------
    scores_per_dataset : dict
        ``{dataset_label: [scores_dict, ...]}`` where each *scores_dict* is
        the dict returned by ``cross_val()`` containing lists of per-fold
        values for every metric.
    dataset_labels : list of str
        Ordered list of keys present in *scores_per_dataset*.
    save : bool
        Whether to save the figures to disk.
    figname : str or None
        Base file path (without extension) used when *save* is True.
        ``'_bar.{png,pdf}'`` and ``'_boxplot.{png,pdf}'`` are appended.

    Returns
    -------
    fig_bar : matplotlib.figure.Figure
    fig_box : matplotlib.figure.Figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'FScore']

    # Aggregate all fold values per metric per dataset (pool across all combos)
    aggregated = {}
    for ds_label in dataset_labels:
        agg = {m: [] for m in metrics}
        for scores_dict in scores_per_dataset[ds_label]:
            for m in metrics:
                agg[m].extend(scores_dict[m])
        aggregated[ds_label] = agg

    n_datasets = len(dataset_labels)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_datasets

    # --- Grouped bar chart ---
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    for i, ds_label in enumerate(dataset_labels):
        means = [np.mean(aggregated[ds_label][m]) for m in metrics]
        stds = [np.std(aggregated[ds_label][m]) * 2 for m in metrics]
        offset = (i - n_datasets / 2 + 0.5) * width
        ax_bar.bar(x + offset, means, width, label=ds_label, yerr=stds, capsize=4)

    ax_bar.set_xlabel('Metric')
    ax_bar.set_ylabel('Score')
    ax_bar.set_title('Dataset Performance Comparison (mean ± 2σ across all classifiers and folds)')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.legend(loc='lower right')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    fig_bar.tight_layout()

    if save and figname:
        fig_bar.savefig(figname + '_bar.png', dpi=fig_bar.dpi, bbox_inches='tight')
        fig_bar.savefig(figname + '_bar.pdf', dpi=fig_bar.dpi, bbox_inches='tight', format='pdf')

    # --- Box plots ---
    fig_box, axes_box = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6), sharey=True)
    if n_metrics == 1:
        axes_box = [axes_box]

    for ax, metric in zip(axes_box, metrics):
        data_for_box = [aggregated[ds_label][metric] for ds_label in dataset_labels]
        ax.boxplot(data_for_box, labels=dataset_labels, patch_artist=True)
        ax.set_title(metric)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig_box.suptitle('Dataset Performance Comparison – Score Distributions across Classifiers and Folds')
    fig_box.tight_layout()

    if save and figname:
        fig_box.savefig(figname + '_boxplot.png', dpi=fig_box.dpi, bbox_inches='tight')
        fig_box.savefig(figname + '_boxplot.pdf', dpi=fig_box.dpi, bbox_inches='tight', format='pdf')

    return fig_bar, fig_box


def plot_pca_scatter_comparison(datasets_dict, dataset_labels, class_names=None, save=False, figname=None):
    """Side-by-side 2D PCA scatter plots for each dataset.

    Uses the first two principal components already stored in the
    ``Combined_Dataset.X`` attribute after ``create_dataset()`` was called.

    Parameters
    ----------
    datasets_dict : dict
        ``{dataset_label: Combined_Dataset_object}`` – each object must expose
        ``.X`` (PCA-transformed, shape ``(n_samples, n_components)``) and
        ``.y`` (integer class labels).
    dataset_labels : list of str
    class_names : list of str or None
        Human-readable class names (e.g. ``['correct', 'wrong']``).
    save : bool
    figname : str or None
        Base file path; ``'.{png,pdf}'`` is appended.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if class_names is None:
        class_names = ['correct', 'wrong']

    n_datasets = len(dataset_labels)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax, ds_label in zip(axes, dataset_labels):
        dataset = datasets_dict[ds_label]
        X = dataset.X
        y = dataset.y

        pc2 = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
        for class_idx, class_name in enumerate(class_names):
            mask = np.array(y) == class_idx
            ax.scatter(
                X[mask, 0],
                pc2[mask],
                label=class_name,
                alpha=0.7,
                s=30,
                color=colors[class_idx % len(colors)]
            )

        ax.set_title(f'PCA Scatter – {ds_label}')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.legend(loc='best')
        ax.grid(linestyle='--', alpha=0.4)

    fig.suptitle('PCA 2D Scatter Comparison across Datasets')
    fig.tight_layout()

    if save and figname:
        fig.savefig(figname + '.png', dpi=fig.dpi, bbox_inches='tight')
        fig.savefig(figname + '.pdf', dpi=fig.dpi, bbox_inches='tight', format='pdf')

    return fig


def plot_confusion_matrices_comparison(predictions_per_dataset, dataset_labels, class_names=None,
                                       save=False, figname=None):
    """Side-by-side normalised confusion-matrix heatmaps for each dataset.

    Parameters
    ----------
    predictions_per_dataset : dict
        ``{dataset_label: (y_pred_all, y_test_all)}`` – both lists contain
        integer class indices aggregated across all cross-validation folds.
    dataset_labels : list of str
    class_names : list of str or None
    save : bool
    figname : str or None
        Base file path; ``'.{png,pdf}'`` is appended.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    if class_names is None:
        class_names = ['correct', 'wrong']

    n_datasets = len(dataset_labels)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for ax, ds_label in zip(axes, dataset_labels):
        y_pred, y_test = predictions_per_dataset[ds_label]
        if len(y_pred) == 0:
            ax.set_title(f'Confusion Matrix – {ds_label}\n(no data)')
            continue

        cm = sk_confusion_matrix(y_test, y_pred)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums == 0, 0, cm.astype(float) / row_sums)

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Confusion Matrix – {ds_label}')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        plt.colorbar(im, ax=ax)

        thresh = 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                        ha='center', va='center',
                        color='white' if cm_norm[i, j] > thresh else 'black')

    fig.suptitle('Confusion Matrices Comparison across Datasets')
    fig.tight_layout()

    if save and figname:
        fig.savefig(figname + '.png', dpi=fig.dpi, bbox_inches='tight')
        fig.savefig(figname + '.pdf', dpi=fig.dpi, bbox_inches='tight', format='pdf')

    return fig

