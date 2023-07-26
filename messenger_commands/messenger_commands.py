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
    # set a figure space
    fig, ax = plt.subplots()
    # get rid of axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # no frame
    ax.set_frame_on(False)
    # turn DataFrame into a table format
    tab = table(ax,
                df,
                loc="upper right",
                colWidths=[0.2] * len(df.columns))
    # set font manually
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)

    # set table size
    tab.scale(2, 2)
    plt.subplots_adjust

    slash = _slash_conversion()
    dir_path = _ROOT_PATH()
    plt.savefig(dir_path + slash + 'vaults' + slash + "picture_vault" + slash + table_name + ".png",
                bbox_inches='tight')
    # _send_telegram_photo("picture_vault/" + table_name + ".png")
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
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
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
    plt.hist(prediction.values.flatten(), bins=100, density=True, alpha=0.5,
             histtype='stepfilled', color='steelblue',
             edgecolor='none')
    plt.ylabel('Probability')
    plt.xlabel('Data Values')
    plt.title('Probability Distribution of Data')

    plt.xlim([0, 1])  # Set limits of x-axis
    plt.axvline(0.5, color='r')  # Add a vertical line at x=0.5, color red

    dir_path = _ROOT_PATH()
    slash = _slash_conversion()
    save_path = dir_path + slash + 'vaults' + slash + 'picture_vault' + slash + 'probability_distribution.png'

    plt.savefig(save_path)
    _send_discord_photo(save_path)


