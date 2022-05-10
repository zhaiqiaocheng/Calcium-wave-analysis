import numpy as np
import pywt
from scipy import signal
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from threading import Thread


def export_to_csv(coefs, frequencies, file='data/exported_data'):
    print('exporting to', file)
    with open(file, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(frequencies)

        for row_no in range(0, len(coefs[0])):
            row_data = []
            for col_no in range(0, len(frequencies)):
                row_data.append(abs(coefs[col_no][row_no]))
            csv_writer.writerow(row_data)

def export_dff_to_csv(dff, file='data/exported_dff'):
    print('exporting dff to', file)
    with open(file, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['\u0394F / F'])
        for d in dff:
            csv_writer.writerow([d])

def calculate_dff_tf_data(signals, sampling_rate):
    print('continuous wavelet transform')
    nd_sig = np.array(signals)
    nd_sig = nd_sig[nd_sig > np.mean(signals)]
    sig_baseline = sum(nd_sig) / sum(nd_sig > 0)

    standard_sig_dff = (signals - sig_baseline) / sig_baseline
    standard_sig_dff = standard_sig_dff + abs(np.min(standard_sig_dff))

    wave_name = 'cgau8'
    if 'Wavelet' in globals().keys():
        wave_name = Wavelet

    total_scale = 128
    if 'FrequencyResolution' in globals().keys():
        total_scale = FrequencyResolution

    print('params', wave_name, total_scale)

    cf = pywt.central_frequency(wave_name)
    scales = 2 * cf * total_scale / np.arange(total_scale, 1, -1)
    [coefs, frequencies] = pywt.cwt(standard_sig_dff, scales, wave_name, 1.0 / sampling_rate)

    return standard_sig_dff, coefs, frequencies


def plot_data(signals, dff, coefs, frequencies, title='Time Frequency Analysis'):
    t = np.arange(0, len(signals))
    sns.set()

    fig, axs = plt.subplots(3, 1, sharex='all', figsize=(9, 5))
    fig.canvas.set_window_title(title)
    fig.subplots_adjust(hspace=0.2)

    axs[0].plot(t, signals)
    axs[0].set_title(title)
    axs[0].set_ylabel('Gray Value', fontsize=10)

    axs[1].plot(t, dff)
    axs[1].set_ylabel('\u0394F / F', fontsize=10)

    z_lim = np.arange(-0.02, 1.0, 0.01)
    if 'ZLimits' in globals().keys():
        z_lim = ZLimits

    ax2 = axs[2].contourf(t, frequencies, abs(coefs), z_lim, cmap="jet")
    axs[2].set_xlabel('Time (S)', fontsize=10)
    axs[2].set_ylabel('Frequency (Hz)', fontsize=10)
    fig.colorbar(ax2, ax=axs, shrink=0.6)

    if 'SaveSVG' in globals().keys() and SaveSVG is True:
        print('saving svg', title + '.svg')
        plt.savefig(title + '.svg', format='svg')

    if 'SavePNG' in globals().keys() and SavePNG is True:
        print('saving png', title + '.png')
        plt.savefig(title + '.png', format='png')

    if 'ShowFigure' in globals().keys() and ShowFigure is True:
        plt.show()


def segmentation(sig, timestamp):
    sig_len = len(sig)
    ts_len = len(timestamp)
    if sig_len != ts_len:
        return []

    r = []
    b_interval = 100
    section = ([sig[0]], [timestamp[0]])
    for i in range(1, sig_len):
        interval = timestamp[i] - timestamp[i - 1]
        if interval > b_interval:
            print('interval:', interval)
            r.append(section)
            section = ([sig[i]], [timestamp[i]])
        else:
            section[0].append(sig[i])
            section[1].append(timestamp[i])
    return r


def load_csv(file_name):
    d = {'sn': [], 'c-1': [], 'c-2': [], 'c-3': [], 'c-4': [], 'ts': []}
    with open(file_name, 'r') as csv_file:
        content_list = list(csv.reader(csv_file))
        # base_time = float(content_list[0][10]) / 1000000000
        for row in content_list:
            d['sn'].append(int(row[0]))
            d['c-1'].append(float(row[1]))
            d['c-2'].append(float(row[2]))
            d['c-3'].append(float(row[3]))
            d['c-4'].append(float(row[4]))
            # d['ts'].append((float(row[10]) / 1000000000) - base_time)
    return d


def guess_sampling_rate(timestamps):
    return 15
    # return 1.0 / ((timestamps[-1] - timestamps[0]) / (len(timestamps) - 1))


def main_process():
    files = [r'D:\data\after3day.csv']
    chns = ['c-1', 'c-2', 'c-3', 'c-4']

    for f in files:
        data = load_csv(f)
        print('read of data:', len(data['c-1']))

        sp = guess_sampling_rate(data['ts'][0:100])
        print('guess sampling rate is', sp)
        for c in chns:
            print('processing ' + f + ' ' + c)
            c_id = f.replace('.csv', '') + '_' + c
            dff, coefs, frequencies = calculate_dff_tf_data(data[c], sp)
            plot_data(data[c], dff, coefs, frequencies, c_id)

            if 'ExportResult' in globals().keys() and ExportResult is True:
                export_to_csv(coefs, frequencies, c_id + '.csv')
                export_dff_to_csv(dff, c_id + '_dff.csv')
    print('All jobs are done.')


if __name__ == '__main__':
    ShowFigure = False

    SaveSVG = True
    SavePNG = True
    ExportResult = True

    Wavelet = 'cgau8'    # 'cgau8' 'morl' etc.
    FrequencyResolution = 14
    ZLimits = np.arange(-0.02, 1.0, 0.01)

    main_process()
