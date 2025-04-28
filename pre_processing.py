import numpy as np
import torch
from libs.utils import normalize_tensor, CustomDataset
from libs.Args import args

def create_sequences(data, seq_length, labels=None):
    sequences = []
    label_sequences = []

    if labels == None:

        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            sequences.append(seq)

        return np.array(sequences)
    else:
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label_seq = labels[i:i + seq_length]
            sequences.append(seq)
            label_sequences.append(label_seq)
        return np.array(sequences), np.array(label_sequences)
def get_data(sequence_length):
    feat_pre = np.genfromtxt('D:/TD_ODE/data/cleaned_imputed_features.csv', delimiter=',', filling_values=np.nan)
    label_m_pre = np.genfromtxt('D:/TD_ODE/data/cleaned_labels_M.csv', delimiter=',', filling_values=np.nan)

    feat = torch.tensor(feat_pre[~np.isnan(feat_pre).any(axis=1)][:, 1:])  # (504,6)
    label_m = torch.tensor(label_m_pre[~np.isnan(label_m_pre).any(axis=1)][:, 1:2])  # extract export(504,1)

    if label_m.shape[0] % args.F != 0:
        raise ValueError("This matrix cannot be aggregated completely, you should slice this matrix")

    reshaped_matrix = label_m.reshape(-1, 4)
    aggregated_matrix = reshaped_matrix.sum(axis=1)
    label_q = aggregated_matrix.reshape(-1, 1)
    time_steps_ode = torch.tensor(feat_pre[~np.isnan(feat_pre).any(axis=1)][:, 0:1])  # (504,1)

    # print(f"Features' shape is {feat.shape}", "\n",
    #       f"Label_Quarter's shape is {label_q.shape}", "\n",
    #       f"Label_Month's shape is {label_m.shape}", "\n",
    #       f"time steps' shape is {time_steps_ode.shape}", "\n")

    feat_nor = normalize_tensor(feat)
    truth_q_nor = normalize_tensor(label_q)
    truth_m_nor = normalize_tensor(label_m)


    feat_seq, truth_m_seq = create_sequences(feat_nor, sequence_length, truth_m_nor)
    truth_q_seq = create_sequences(truth_q_nor,sequence_length)

    return feat_seq, label_q, truth_m_seq, label_m, truth_q_seq, truth_q_nor, feat_nor




if __name__ == "__main__":
    get_data(4)