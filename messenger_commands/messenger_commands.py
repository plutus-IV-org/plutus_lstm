import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import time
from PATH_CONFIG import _ROOT_PATH
from utilities.service_functions import _discord_channel, _slash_conversion

plt.style.use('ggplot')


def _message_decorator(func):
    def inner(str):
        key = True
        while key == True:
            try:
                func(str)
                key = False
            except:
                time.sleep(1)

    return inner


import requests


@_message_decorator
def _send_discord_message(bot_msg):
    channelID = _discord_channel()
    botToken = "MTA1MTUzODk4NDk4NTE4NjQ4NA.G0nfjy.3q_DDES9KzAI5rmNjaFvEoJsuSc33Se2-4FJpE"
    baseURL = "https://discordapp.com/api/channels/{}/messages".format(channelID)
    headers = {"Authorization": "Bot {}".format(botToken)}
    message = {
        "content": bot_msg
    }
    requests.post(baseURL, headers=headers, data=message)


# @_message_decorator
def _send_discord_photo(photo_name):
    channelID = _discord_channel()
    botToken = "MTA1MTUzODk4NDk4NTE4NjQ4NA.G0nfjy.3q_DDES9KzAI5rmNjaFvEoJsuSc33Se2-4FJpE"
    baseURL = "https://discordapp.com/api/channels/{}/messages".format(channelID)
    headers = {"Authorization": "Bot {}".format(botToken)}
    dir_containing_file = _ROOT_PATH()
    slash = _slash_conversion()
    files = {
        "file": (photo_name,
                 open(photo_name, 'rb'))
    }
    requests.post(baseURL, headers=headers, files=files)


@_message_decorator
def _send_telegram_msg(bot_msg):
    token_id = "5696424279:AAFgJGJkOLf76N--Ur2b0OMmrUXPx2ZJzgs"
    chat_id = "-1001545003820"

    bot_msg = bot_msg.replace('-', ' ')
    bot_msg = bot_msg.replace('=', ' ')

    send_text = "https://api.telegram.org/bot" + token_id + "/sendMessage?chat_id=" + chat_id + "&parse_mode=MarkdownV2&text=" + bot_msg
    response = requests.get(send_text)
    return response.json()


@_message_decorator
def _send_telegram_photo(photo_name):
    img = open(photo_name, 'rb')
    TOKEN = "5696424279:AAFgJGJkOLf76N--Ur2b0OMmrUXPx2ZJzgs"
    CHAT_ID = "-1001545003820"
    url = f'https://api.telegram.org/bot{TOKEN}/sendPhoto?chat_id={CHAT_ID}'
    print(requests.post(url, files={'photo': img}))


def _dataframe_to_png(df, table_name):
    """
    Convert a DataFrame to a PNG image with a title.

    Parameters:
    - df: pd.DataFrame, the data to render as a table.
    - table_name: str, the title to be added at the top of the plot.
    """
    # Set a figure space
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust figsize as needed

    # Add the table name as a title with reduced padding
    plt.title(table_name, fontsize=14, fontweight='bold', pad=2)  # Significantly reduced padding

    # Get rid of axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # No frame
    ax.set_frame_on(False)

    # Turn DataFrame into a table format
    tab = table(ax,
                df,
                loc="center",  # Adjust location
                colWidths=[0.2] * len(df.columns))

    # Set font manually
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)

    # Set table size
    tab.scale(2, 2)

    # Save the table as an image
    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    plt.savefig(dir_path + slash + 'vaults' + slash + "picture_vault" + slash + table_name + ".png",
                bbox_inches='tight')

    # Send to Discord (optional)
    _send_discord_photo(dir_path + slash + 'vaults' + slash + "picture_vault" + slash + table_name + ".png")


def _visualize_loss_results(results):
    plt.rcParams.update({'axes.facecolor': 'white'})
    history = results.history
    fig = plt.figure(figsize=(14, 8))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    fig.savefig(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'loss.png')
    # _send_telegram_photo('picture_vault' + slash + 'loss.png')
    _send_discord_photo(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'loss.png')


def _visualize_mda_results(results):
    history = results.history
    fig = plt.figure(figsize=(14, 8))
    plt.plot(history['val_mda'])
    plt.plot(history['mda'])
    plt.legend(['val_mda', 'mda'])
    plt.title('mda')
    plt.xlabel('epochs')
    plt.ylabel('mda')
    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    fig.savefig(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'mda.png')
    # _send_telegram_photo('picture_vault' + slash + 'accuracy.png')
    _send_discord_photo(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'mda.png')


def _visualize_accuracy_results(results):
    history = results.history
    fig = plt.figure(figsize=(14, 8))
    metrics = [key for key in history.keys() if 'loss' not in key]
    for metric in metrics:
        plt.plot(history[metric], label=metric)
    plt.legend(metric)
    plt.title('metric_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('metric')
    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    fig.savefig(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'accuracy.png')
    # _send_telegram_photo('picture_vault' + slash + 'accuracy.png')
    _send_discord_photo(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'accuracy.png')


def _visualize_prediction_results_daily(prediction, actual):
    df_p = pd.DataFrame(prediction)
    df_a = pd.DataFrame(actual)
    for i in range(0, df_p.shape[1]):
        d = df_a[i] - df_p[i]
        g = abs(d).copy()
        y = abs(d).copy()

        g[g > 0.01] = 0
        g[g != 0] = 1
        y[y > 0.025] = 0
        y[y != 0] = 1

        ga = round(g.mean(), 2)
        ya = (round(y.mean(), 2) - ga) * 100
        ga = ga * 100
        ra = round(1 - y.mean(), 2) * 100

        fig = plt.figure(figsize=(40, 15))
        plt.stem(d.index, abs(d.values))
        plt.xlim(-1, int(d.index[-1]) + 1)
        plt.axhline(y=0.01, color='g', linestyle='dashed')
        plt.axhline(y=0.025, color='y', linestyle='dashed')
        plt.title('Absolute Actual-Predicted Diff Day ' + str(i + 1))
        plt.ylabel('Price')
        plt.legend([f'Green - {ga}%, Yellow - {ya}%,Red - {ra}%'])
        slash = _slash_conversion()
        dir_path = _ROOT_PATH()
        fig.savefig(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'prediction' + str(i) + '.png')
        # _send_telegram_photo('picture_vault' + slash + 'prediction' + str(i) + '.png')
        _send_discord_photo(
            dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'prediction' + str(i) + '.png')


def _visualize_prediction_results(prediction, actual):
    df_p = pd.DataFrame(prediction)
    df_a = pd.DataFrame(actual)
    d = df_a - df_p
    fig, ax = plt.subplots(figsize=(40, 15))
    ind = d.index.tolist() * len(df_p.columns)
    val = d.values.reshape(-1, 1)
    ind.sort()
    if len(df_p.columns) > 7:
        ax.stem(ind, abs(val))
    else:
        colors = ['k', 'r', 'y', 'g', 'c', 'b', 'm']
        for x in reversed(range(len(df_p.columns))):
            name = 'Lag ' + str(x + 1)
            mak = colors[x] + 'o'
            ax.stem(ind[x::len(df_p.columns)], abs(val[x::len(df_p.columns)]), colors[x], markerfmt=mak, basefmt=" ",
                    label=name)

    plt.xlim(-1, int(d.index[-1]) + 1)
    plt.axhline(y=0.01, color='g', linestyle='dashed')
    plt.axhline(y=0.025, color='y', linestyle='dashed')
    plt.title('Absolute Actual-Predicted Diff')
    plt.ylabel('Price')
    plt.legend()
    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    fig.savefig(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'prediction.png')
    # _send_telegram_photo('picture_vault' + slash + 'prediction.png')
    _send_discord_photo(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'prediction.png')


def _visualize_probability_distribution(prediction):
    # Create the figure
    plt.figure(figsize=(10, 6))

    # Iterate over each column in the prediction DataFrame
    for col_idx, column in enumerate(prediction.columns):
        plt.hist(
            prediction[column].values.flatten(),
            bins=100,
            density=True,
            alpha=0.5,
            histtype='stepfilled',
            edgecolor='none',
            label=f'Lag {col_idx + 1}',
            linewidth=1.5
        )

    # Add labels and title
    plt.ylabel('Probability')
    plt.xlabel('Data Values')
    plt.title('Probability Distribution of Data')

    # Set limits of x-axis
    plt.xlim([0, 1])

    # Add a vertical line at x=0.5
    plt.axvline(0.5, color='r', linestyle='--', linewidth=1.2)

    # Add legend
    plt.legend(title='Lag Predictions', loc='upper right')

    # Save the plot
    dir_path = _ROOT_PATH()
    slash = _slash_conversion()
    save_path = f"{dir_path}{slash}vaults{slash}picture_vault{slash}probability_distribution.png"

    # Save and send the plot to Discord (or any platform you use)
    plt.savefig(save_path)
    _send_discord_photo(save_path)


def _visualize_cross_validation_loss_results(results_list):
    """
    Visualize loss from a list of training histories with vertical lines for each fold.
    Adjust y-axis to focus on the main range of values, ignoring initial peaks.

    Parameters:
    - results_list: list of history objects from each fold.
    """
    plt.style.use('default')  # Reset to default for white background
    fig, ax = plt.subplots(figsize=(14, 8))
    epoch_counter = 0  # Track cumulative epochs

    # Collect loss and validation loss across all folds
    combined_loss = []
    combined_val_loss = []

    for i, fold_history in enumerate(results_list):
        loss = fold_history['loss']
        val_loss = fold_history['val_loss']

        # Extend combined metrics and count epochs
        combined_loss.extend(loss)
        combined_val_loss.extend(val_loss)
        epochs = range(epoch_counter, epoch_counter + len(loss))

        # Plot the metrics for this fold
        ax.plot(epochs, loss, color='blue', label='Training Loss' if i == 0 else "")
        ax.plot(epochs, val_loss, color='red', label='Validation Loss' if i == 0 else "")

        # Add a vertical line at the end of each fold
        ax.axvline(x=epoch_counter + len(loss) - 1, color='gray', linestyle='--', linewidth=0.5)
        epoch_counter += len(loss)

    # Set y-limits based on the 1st and 99th percentiles
    all_values = combined_loss + combined_val_loss
    lower_limit = np.percentile(all_values[5:], 1) - 0.01
    upper_limit = np.percentile(all_values[5:], 99)
    ax.set_ylim(lower_limit, upper_limit)

    ax.legend(title="Loss Types", loc="upper right")
    ax.set_title('Loss Across Folds')
    ax.set_xlabel('Epochs (cumulative across folds)')
    ax.set_ylabel('Loss')

    # Save the figure
    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    fig.savefig(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'loss.png')
    _send_discord_photo(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'loss.png')


def _visualize_cross_validation_accuracy_results(results_list):
    """
    Visualize accuracy-related metrics from a list of training histories with vertical lines for each fold.
    Adjust y-axis to focus on the main range of values, ignoring initial spikes.

    Parameters:
    - results_list: list of history objects from each fold.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))
    epoch_counter = 0  # Track cumulative epochs

    # Identify accuracy metrics by excluding loss values
    metric_keys = [key for key in results_list[0] if 'loss' not in key]

    # Initialize combined lists for each metric
    combined_metrics = {key: [] for key in metric_keys}

    for i, fold_history in enumerate(results_list):
        # Collect values for each metric
        for metric in metric_keys:
            metric_values = fold_history[metric]
            combined_metrics[metric].extend(metric_values)

            # Plot the metric values with consistent colors across folds
            epochs = range(epoch_counter, epoch_counter + len(metric_values))
            ax.plot(epochs, metric_values, label=f"{metric}" if i == 0 else "",
                    color='blue' if 'val' not in metric else 'orange')

        # Add a vertical line at the end of each fold
        ax.axvline(x=epoch_counter + len(metric_values) - 1, color='gray', linestyle='--', linewidth=0.5)
        epoch_counter += len(metric_values)

    # Set y-limits based on the 5th and 95th percentiles of combined metrics
    all_values = [val for metric_values in combined_metrics.values() for val in metric_values]
    lower_limit = np.percentile(all_values, 1) - 0.01
    upper_limit = np.percentile(all_values, 99)
    ax.set_ylim(lower_limit, upper_limit)

    ax.legend(title="Metrics")
    ax.set_title('Accuracy Metrics Across Folds')
    ax.set_xlabel('Epochs (cumulative across folds)')
    ax.set_ylabel('Metric Value')

    # Save the figure
    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    fig.savefig(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'accuracy.png')
    _send_discord_photo(dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'accuracy.png')


def save_figure(fig, name):
    """
    Helper function to save figures and send via Discord.

    Parameters:
    - fig: The figure to save
    - name: The name prefix for the saved file
    """
    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    save_path = f"{dir_path}{slash}vaults{slash}picture_vault{slash}{name}.png"
    fig.savefig(save_path)
    _send_discord_photo(save_path)
