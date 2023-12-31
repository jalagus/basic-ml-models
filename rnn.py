import torch
import torch.nn as nn
import numpy as np


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size))
        self.bias_h = nn.Parameter(torch.zeros(1, hidden_size))
        self.bias_y = nn.Parameter(torch.zeros(1, output_size))
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state
        h_t = torch.zeros(batch_size, self.hidden_size)
        
        for t in range(seq_length):
            x_t = x[:, t, :]            
            h_t = torch.tanh(torch.mm(x_t, self.Wxh) + torch.mm(h_t, self.Whh) + self.bias_h)
            
        y_t = torch.mm(h_t, self.Why) + self.bias_y
        return y_t
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size))
        self.bias_y = nn.Parameter(torch.zeros(1, output_size))

        self.Wf = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Uf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_f = nn.Parameter(torch.zeros(1, hidden_size))

        self.Wi = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Ui = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_i = nn.Parameter(torch.zeros(1, hidden_size))

        self.Wc = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Uc = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_c = nn.Parameter(torch.zeros(1, hidden_size))

        self.Wo = nn.Parameter(torch.randn(input_size, hidden_size))
        self.Uo = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_o = nn.Parameter(torch.zeros(1, hidden_size))


    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state
        h_t = torch.zeros(batch_size, self.hidden_size)
        C_t = torch.zeros(batch_size, self.hidden_size)
        
        for t in range(seq_length):
            x_t = x[:, t, :]

            f_t = torch.sigmoid(torch.mm(x_t, self.Wf) + torch.mm(h_t, self.Uf) + self.bias_f)
            i_t = torch.sigmoid(torch.mm(x_t, self.Wi) + torch.mm(h_t, self.Ui) + self.bias_i)
            o_t = torch.sigmoid(torch.mm(x_t, self.Wo) + torch.mm(h_t, self.Uo) + self.bias_o)
            Ch_t = torch.sigmoid(torch.mm(x_t, self.Wc) + torch.mm(h_t, self.Uc) + self.bias_c)

            C_t = f_t * C_t + i_t * Ch_t
            h_t = o_t * torch.tanh(C_t)
            
        y_t = torch.mm(h_t, self.Why) + self.bias_y
        return y_t

# Simple demo
if __name__ == "__main__":
    dataset = []
    for i in range(100):
        dataset.append(np.linspace(i, i + 10, 11) / 10)

    dataset = torch.tensor(np.array(dataset), dtype=torch.float32)
    normalized_data = (dataset - dataset.mean()) / dataset.std()

    labels = normalized_data[:,-1, None]
    train_dataset = normalized_data[:,:-1,None]

    # Init the RNN for single input and single output
    model = LSTM(1, 50, 1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(train_dataset)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    with torch.inference_mode():
        # Predict outside training domain
        x = 10
        test_instance = torch.tensor(np.linspace(x, x + 10, 11) / 10, dtype=torch.float32) 
        print(test_instance)
        norm_test_instance = (test_instance - dataset.mean()) / dataset.std()
        pred_labels = model(norm_test_instance[None,:10,None])

        print("Predicted labels:")
        print((pred_labels * dataset.std() + dataset.mean()).flatten())
        print(f"Expected: {test_instance[-1]}")
