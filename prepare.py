import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks  
from imblearn.over_sampling import SMOTE
import torch, os

class DataProcessing:
    def __init__(self, filenames, lb, skip_lines=15):
        self.filenames = filenames # list of filenames 
        self.skip_lines = skip_lines
        # raw data
        self.time, self.voltage, self.current = [], [], []
        # cleaned data holders
        self.clean_v, self.clean_dv, self.clean_dvv = None, None, None
        # labels and feats
        self.labels, self.features, self.stft_time = None, None, None
        # set lower bound for labeling
        self.lb = lb

    def load_files(self):
        """
        list of filenames -> loaded time, voltage, curr data
        """
        for f in self.filenames:
            with open(f) as f:
                for i in range(self.skip_lines): # skip meta data
                    f.readline()
                for line in f:
                    values = line.strip().split() # split ',' or comma separated stuff
                    # convert values to float 
                    self.time.append(float(values[0]))
                    self.voltage.append(float(values[1]))
                    self.current.append(float(values[2]))
            print("length of time:", len(self.time), len(self.voltage), len(self.current))
        # convert to numpy once after all files are loaded
        self.time = np.array(self.time)
        self.voltage = np.array(self.voltage)
        self.current = np.array(self.current)
    
    def compute_diffs(self):
        """
        compute finite diffs -> dv
        """
        dv = self.voltage[1:] - self.voltage[:-1]
        return dv

    def find_peaks(self, data, min_height=2, min_distance=500):
        """
        find indices of peaks 
        """
        peaks, _ = find_peaks(data, height=min_height, distance=min_distance)
        return peaks

    def clean_data(self):
        """
        extract +/- 35 ms around each peak
        """
        dv = self.compute_diffs()
        sd = np.std(dv[:500]) # compute s.d. for baseline
        noisy_maxima = self.find_peaks(dv, min_height=2, min_distance=500)
        # copy data
        v_clean = self.voltage.copy()
        dv_clean = dv.copy()
        # initialise clean holders
        cleaned_v, cleaned_dv = [], []
        half_win = 350
        for idx in noisy_maxima:
            cleaned_v.extend(v_clean[idx - half_win: idx + half_win])
            cleaned_dv.extend(dv_clean[idx - half_win: idx + half_win])

        self.clean_v = np.array(cleaned_v)
        self.clean_dv = np.array(cleaned_dv)
        self.sd = sd

    def label_data(self, window=300, ub=13):
        """
        label data using a window approach
        """
        sd = self.sd 
        labels = []
        for i in range(0, len(self.clean_dv) - window):
            segment_std = np.std(self.clean_dv[i:i+window])
            if self.lb * sd < segment_std < ub*sd:
                labels.append(1)
            else: 
                labels.append(0)
        self.labels = np.array(labels)
    
    def find_rising_edges(self, arr):
        """
        return indices where labels switch from 0->1 or 1->0
        """
        arr = np.array(arr)
        return [i for i in range(1, len(arr)) if arr[i-1]==0 and arr[i]==1]

    def get_stft(self, signal, st, window_size, incr, last_sig):
        """
        stft calc
        """
        fft_result = []
        fft_time = []
        en = st+window_size
        while en <= last_sig:
            window = signal[st:en]
            fft = np.abs(np.fft.fft(window)[: window_size//2])
            fft_result.append(fft)
            st += incr
            fft_time.append(en)
            en = st+window_size
        return np.array(fft_result), np.array(fft_time)

    def build_features(self,
                       total_length=20000,
                       length=500,
                       window=100,
                       incr=50,
                       step=10):
        """
        get stft_v, stft_dv, fft_labels 
        """
        fv = self.clean_v[:total_length]
        fdv = self.clean_dv[:total_length]
        flabel = self.labels[:total_length]

        def count_dominant(arr):
            zeros = len(arr) - np.count_nonzero(arr)
            ones = np.count_nonzero(arr)
            return 0 if zeros > ones else 1

        stft_v, stft_dv, stft_time, fft_label = [], [], [], []
        for st in range(0, total_length - length, step):
            last_sig = st + length
            fft_v, _ = self.get_stft(fv, st, window, incr, last_sig)
            fft_dv, _ = self.get_stft(fdv, st, window, incr, last_sig)

            stft_v.append(fft_v)
            stft_dv.append(fft_dv)
            stft_time.append(_)
            fft_label.append(count_dominant(flabel[st:last_sig]))

        self.stft_v = stft_v
        self.stft_dv = stft_dv
        self.stft_time = stft_time
        self.fft_label = fft_label

        X = np.array(stft_dv)
        n_samples, n_timesteps, n_feats = X.shape
        X_flat = X.reshape(n_samples, n_timesteps * n_feats)
        self.features = X_flat
    
    def get_train_test_tensors(self, train_size=1500, test_split=0.2):
        X = self.features
        y = self.fft_label

        X_train, X_test, y_train, y_test = train_test_split(
            X[:train_size], y[:train_size], test_size=test_split, random_state=50
        )
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Convert to torch tensors
        self.X_train = torch.tensor(X_train_res, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train_res, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def visualise(self):
        """
        visualise the data processing pipeline
        """
        self.load_files()
        dv = self.compute_diffs()

        # plot pre-cleaned data
        plt.figure(figsize=(10, 4))
        plt.plot(dv, label='raw dv')
        plt.title('raw dv')
        plt.legend()
        plt.show()  

        # now clean and plot
        self.clean_data()
        print("clean_dv length:", len(self.clean_dv), '\n\n')
        noisy_maxima = self.find_peaks(dv, min_height=2, min_distance=500)
        print("number of peaks:", len(noisy_maxima), '\n\n')
        # get labels
        self.label_data()

        # plot cleaned data + labels
        plt.figure(figsize=(10, 4))
        plt.plot(self.clean_dv, label='clean dv')
        plt.plot(self.labels, label='clean dv')
        plt.title('cleaned dv around peaks')
        plt.legend()
        plt.show()

        # get stfts
        self.build_features(total_length=20000, length=500, window=100, incr=50, step=10)
        # plot stfts
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        for idx in range(40, 100, 20):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            plt.rc('font', family='serif')
            
            # common y‐tick indices and labels
            y_inds = np.arange(0, len(self.stft_time[idx]), 2)
            y_labels = [int(self.stft_time[idx][i] / 10) for i in y_inds]
            
            def plot_panel(ax, data):
                im = ax.imshow(data[:, :8])
                ax.set_xticks(np.arange(8))
                ax.set_yticks(y_inds)
                ax.set_yticklabels(y_labels)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_xlabel('Windows', fontsize=18)
                ax.set_ylabel('Time (ms)', fontsize=18)
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cax.tick_params(axis='both', which='major', labelsize=18)
                plt.colorbar(im, cax=cax)

            # first subplot: stft_v
            plot_panel(ax1, self.stft_v[idx])
            # second subplot: stft_dv
            plot_panel(ax2, self.stft_dv[idx])

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.8)
            plt.show()

# filenames
# train_folder = 'data/train'

# entries = os.listdir(train_folder)
# train_filenames = [
#     os.path.join(train_folder, fname)
#     for fname in entries
#     if os.path.isfile(os.path.join(train_folder, fname))
# ]

# prep = DataProcessing(train_filenames, lb=4)
# prep.visualise()