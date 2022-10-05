import MorseGen
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 

morse_gen = MorseGen.Morse()
alphabet = morse_gen.alphabet36
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create string for dataset
def get_training_samples():
    return MorseGen.get_morse_str(nchars=132*6, nwords=27*6, chars=alphabet)

# Get database with morse code parameter
def get_new_data(morse_gen, SNR_dB=-23, nchars=132*6, nwords=27*6, phrase=None, alphabet="ABC"):
    if not phrase:
        phrase = MorseGen.get_morse_str(nchars=nchars, nwords=nwords, chars=alphabet)
    Fs = 6000
    samples_per_dit = morse_gen.nb_samples_per_dit(Fs, 20)
    n_prev = int((samples_per_dit/128)*55) + 1 
    label_df = morse_gen.encode_df_decim_tree(phrase, samples_per_dit, 128, alphabet)
    envelope = label_df['env'].to_numpy()
    label_df.drop(columns=['env'], inplace=True)
    SNR_linear = 10.0**(SNR_dB/10.0)  
    SNR_linear *= 256 # Apply original FFT
    t = np.linspace(0, len(envelope)-1, len(envelope))
    power = np.sum(envelope**2)/len(envelope)
    print(power)
    noise_power = power/SNR_linear
    noise = np.sqrt(noise_power)*np.random.normal(0, 1, len(envelope))
    # noise = butter_lowpass_filter(raw_noise, 0.9, 3) # Noise is also filtered in the original setup from audio. This empirically simulates it
    signal = (envelope + noise)**2
    #signal[signal > 1.0] = 1.0 # a bit crap ...
    return envelope, signal, label_df, n_prev

# Define class for morse dataset
class MorsekeyingDataset(torch.utils.data.Dataset):
    def __init__(self, morse_gen, device, SNR_dB=-23, nchars=132, nwords=27, phrase=None, alphabet="ABC"):
        self.envelope, self.signal, self.label_df0, self.seq_len = get_new_data(morse_gen, SNR_dB=SNR_dB, phrase=phrase, alphabet=alphabet)
        self.label_df = self.label_df0.drop(columns=['dit','dah'])
        self.X = torch.FloatTensor(self.signal).to(device)
        self.y = torch.FloatTensor(self.label_df.values).to(device)
        
    def __len__(self):
        return self.X.__len__() - self.seq_len

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.y[index+self.seq_len])
    
    def get_envelope(self):
        return self.envelope
    
    def get_signal(self):
        return self.signal
    
    def get_X(self):
        return self.X
    
    def get_labels(self):
        return self.label_df
    
    def get_labels0(self):
        return self.label_df0
    
    def get_seq_len(self):
        return self.seq_len()


# Get traning dataset from morse generate string 
def get_training_dataset(morsestr, SNR):
    return MorsekeyingDataset(morse_gen, device, SNR, 132*5, 27*5, morsestr, alphabet)

def get_Concat_training_dataset():
    SNR = [-21, -20, -19, -18, -17, -16, -15, -10]
    train_chr_dataset = []
    for value in SNR:
        morse_str = get_training_samples()
        train_chr_dataset0 = get_training_dataset(morse_str, value)
        train_chr_dataset = torch.utils.data.ConcatDataset([train_chr_dataset, train_chr_dataset0])
    return train_chr_dataset

def get_Concat_training_loader(train_chr_dataset):
    return torch.utils.data.DataLoader(train_chr_dataset, batch_size=1, shuffle=True) # Batch size must be 1

class MorseBatchedMultiLSTM(nn.Module):
    """
    Initial implementation
    """
    def __init__(self, device, input_size=1, hidden_layer1_size=6, output1_size=6, hidden_layer2_size=12, output_size=14):
        super().__init__()
        self.device = device # This is the only way to get things work properly with device
        self.input_size = input_size
        self.hidden_layer1_size = hidden_layer1_size
        self.output1_size = output1_size
        self.hidden_layer2_size = hidden_layer2_size
        self.output_size = output_size
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_layer1_size)
        self.linear1 = nn.Linear(hidden_layer1_size, output1_size)
        self.hidden1_cell = (torch.zeros(1, 1, self.hidden_layer1_size).to(self.device),
                             torch.zeros(1, 1, self.hidden_layer1_size).to(self.device))
        self.lstm2 = nn.LSTM(input_size=output1_size, hidden_size=hidden_layer2_size)
        self.linear2 = nn.Linear(hidden_layer2_size, output_size)
        self.hidden2_cell = (torch.zeros(1, 1, self.hidden_layer2_size).to(self.device),
                             torch.zeros(1, 1, self.hidden_layer2_size).to(self.device))
    
    def _minmax(self, x):
        x -= x.min(0)[0]
        x /= x.max(0)[0]
        
    def _hardmax(self, x):
        x /= x.sum()
        
    def _sqmax(self, x):
        x = x**2
        x /= x.sum()
        
    def forward(self, input_seq):
        #print(len(input_seq), input_seq.shape, input_seq.view(-1, 1, 1).shape)
        lstm1_out, self.hidden1_cell = self.lstm1(input_seq.view(-1, 1, self.input_size), self.hidden1_cell)
        pred1 = self.linear1(lstm1_out.view(len(input_seq), -1))
        lstm2_out, self.hidden2_cell = self.lstm2(pred1.view(-1, 1, self.output1_size), self.hidden2_cell)
        predictions = self.linear2(lstm2_out.view(len(pred1), -1))
        self._sqmax(predictions[-1])
        return predictions[-1]
    
    def zero_hidden_cell(self):
        self.hidden1_cell = (
            torch.zeros(1, 1, self.hidden_layer1_size).to(device),
            torch.zeros(1, 1, self.hidden_layer1_size).to(device)
        )     
        self.hidden2_cell = (
            torch.zeros(1, 1, self.hidden_layer2_size).to(device),
            torch.zeros(1, 1, self.hidden_layer2_size).to(device)
        )   

if __name__ == "__main__":
    train_chr_dataset = get_Concat_training_dataset()
    train_chr_loader = get_Concat_training_loader(train_chr_dataset)

    batch_size = 1
    epochs = 4

    morse_chr_model = MorseBatchedMultiLSTM(device, hidden_layer1_size=12, output1_size=4, hidden_layer2_size=len(alphabet)*2, output_size=len(alphabet)+4).to(device) # This is the only way to get things work properly with device
    morse_chr_loss_function = nn.MSELoss()
    morse_chr_optimizer = torch.optim.Adam(morse_chr_model.parameters(), lr=0.001)

    morse_chr_model.train()

    for i in range(epochs):
        train_losses = []
        loop = tqdm(enumerate(train_chr_loader), total=len(train_chr_loader), leave=True)
        for j, train in loop:
            X_train = train[0][0]
            y_train = train[1][0]
            # Xóa Gradient
            morse_chr_optimizer.zero_grad()
            if morse_chr_model.__class__.__name__ in ["MorseLSTM", "MorseLSTM2", "MorseBatchedLSTM", "MorseBatchedLSTM2", "MorseBatchedMultiLSTM", "MorseBatchedLSTMLin2"]:
                morse_chr_model.zero_hidden_cell() # this model needs to reset the hidden cell
            y_pred = morse_chr_model(X_train)
            # Tính toán lỗi và thực hiện backpropagation
            single_loss = morse_chr_loss_function(y_pred, y_train)
            single_loss.backward()
            # Cập nhật trọng số
            morse_chr_optimizer.step()
            train_losses.append(single_loss.item())
            # update progress bar
            if j % 1000 == 0:
                loop.set_description(f"Epoch [{i+1}/{epochs}]")
                loop.set_postfix(loss=np.mean(train_losses))

    print(f'final: {i+1:3} epochs loss: {np.mean(train_losses):6.4f}')

    torch.save(morse_chr_model.state_dict(), '/model/morse_a26_model.pt') 