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
    model = SimpleRNN(1, 50, 1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(train_dataset)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    with torch.inference_mode():
        # Predict outside training domain
        x = 9
        test_instance = torch.tensor(np.linspace(x, x + 10, 11) / 10, dtype=torch.float32) 
        print(test_instance)
        norm_test_instance = (test_instance - dataset.mean()) / dataset.std()
        pred_labels = model(norm_test_instance[None,:10,None])

        print("Predicted labels:")
        print((pred_labels * dataset.std() + dataset.mean()).flatten())
        print(f"Expected: {test_instance[-1]}")
