import logging
import os
import time
import sys
import atexit

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adamax import Adamax
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class GRUNet(nn.Module):
    """
    A GRU model for sequence processing.

    Parameters
    ----------
    input_size : int
        The number of input features (dimension of the input data).
    hidden_size : int
        The number of features in the hidde
        n state of the GRU.
    num_layers : int
        The number of stacked GRU layers.
    output_size : int
        The number of output features (dimension of the final output).
        ............................
    bias : bool, optional, default=True
        Whether to use bias in the GRU layers.
    output_type : {'last', 'mean'}, optional, default='last'
        Determines how to process GRU outputs:
        - 'last' uses the output from the last time step.
        - 'mean' uses the average of all time steps.
    dropout : float, optional, default=0.2
        Dropout rate applied to the GRU layers to prevent overfitting.
    """

    def __init__(self, input_size, xx_size, hidden_size, num_layers, output_size, bias=True, output_type='last', dropout=0.2, seed=1234):
        super(GRUNet, self).__init__()

        # Initialize：Linear+batch_norm
        self.fc1 = nn.Linear(input_size, xx_size)
        self.batch_norm = nn.BatchNorm1d(xx_size)

        # Initialize the GRU layer
        self.gru = nn.GRU(
            input_size = xx_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected layer (adjusted for bidirectional GRU)
        self.fc = nn.Linear(hidden_size, output_size)
        # Output type: 'last' for last timestep or 'mean' for the mean of all timesteps
        self.output_type = output_type

        # Set the random seed for reproducibility
        torch.manual_seed(seed)
        self.initialize_weights()

    def initialize_weights(self):
        # Apply orthogonal initialization for fc1
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)  # Initialize bias to zero

        # Apply orthogonal initialization for GRU's weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)  # Initialize bias to zero

        # Apply orthogonal initialization for fc
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)  # Initialize bias to zero

    def forward(self, x, h_0=None):
        """
        Perform a forward pass through the GRU model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).
        h_0 : torch.Tensor, optional
            Initial hidden state of shape (batch_size, seq_len, input_size).
            If None, it will be initialized to zeros.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the GRU and the fully connected layer.
            The shape of the output is (batch_size, output_size).
        """
        # Pass input through the linear+bach_norm
        x = self.fc1(x)
        x = x.view(x.size(0), x.size(2), x.size(1))
        x = self.batch_norm(x)
        x = x.view(x.size(0), x.size(2), x.size(1))
        # Pass input through the GRU layer
        out, _ = self.gru(x, h_0)

        # Process the output based on the specified output type
        if self.output_type == 'last':
            out = out[:, -1, :]  # Use the output from the last timestep
        elif self.output_type == 'mean':
            out = out.mean(dim=1)  # Compute the mean over all timesteps

        # Pass the processed output through the fully connected layer
        out = self.fc(out)
        return out

def setup_logger(file_name):
    logger = logging.getLogger("time_series_model")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console 输出：兼容 notebook 的 stdout
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)

        # File 输出
        fh = logging.FileHandler(file_name, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        atexit.register(logging.shutdown)

    return logger

def load_data(file_name, seq_len, test_ratio, val_ratio, seed, logger, data_file="processed_data.pt", force_reload=False):
    """
    Load and process time series data using a sliding window, with optional caching to avoid repeated splitting.

    Parameters:
    -----------
    file_name: str
        Path to the raw CSV file. The last column is assumed to be the target variable.
    seq_len: int
        Length of the sliding window (number of time steps used as input).
    test_ratio: float
        Proportion of data to be used for the test set (e.g., 0.2 = 20%).
    val_ratio: float
        Proportion of data to be used for the validation set (e.g., 0.1 = 10%).
    seed: int
        Random seed to ensure reproducibility of data splits.
    data_file: str
        Path to save/load the processed dataset. If it exists, data will be loaded from this file unless force_reload is True.
    force_reload: bool
        If True, the data will be reprocessed and resaved even if cache_file exists.

    Returns:
    --------
    tuple:
        X_train (Tensor): Training input tensor of shape (samples, seq_len, features)
        y_train (Tensor): Training target tensor of shape (samples)
        X_val (Tensor): Validation input tensor
        y_val (Tensor): Validation target tensor
        X_test (Tensor): Test input tensor
        y_test (Tensor): Test target tensor
    """
    # If cached file exists and force_reload is False, load from cache
    if os.path.exists(data_file) and not force_reload:
        logger.info(f"Loading cached data from {data_file}")
        data = torch.load(data_file)
        return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

    # Read raw CSV data
    logger.info(f"Processing data from {file_name}")
    data = pd.read_csv(file_name)
    print(data)
    X_raw = data.iloc[:, :-1].values  # All columns except the last one are features
    y_raw = data.iloc[:, -1].values  # Last column is the target

    # Create sliding windows
    X_windowed, y_windowed = [], []
    for i in range(0, len(X_raw) - seq_len, 1):
        X_windowed.append(X_raw[i:i + seq_len])
        y_windowed.append(y_raw[i + seq_len])  # Predict next step after the window

    # Convert to PyTorch tensors
    X = torch.tensor(X_windowed, dtype=torch.float32)
    y = torch.tensor(y_windowed, dtype=torch.float32)

    # First split: Train vs (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_ratio + val_ratio), random_state=seed, shuffle=True
    )

    # Second split: Validation vs Test
    val_size = val_ratio / (test_ratio + val_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), random_state=seed, shuffle=True
    )

    # Save the processed data to cache
    torch.save({
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }, data_file)
    logger.info(f"Saved processed dataset to {data_file}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def initialize_model(config, device):
    """
    Initialize the model, loss function, and optimizer.

    Parameters
    ----------
    config : dict
        Hyperparameter dictionary containing settings such as input_size, hidden_size, etc.
    device : torch.device
        Device to run the model on ('cuda' or 'cpu').

    Returns
    -------
    model : GRUNet
        The initialized GRU model.
    criterion : nn.Module
        The loss function (Mean Squared Error in this case).
    optimizer : torch.optim.Optimizer
        The optimizer (Adam in this case).
    """
    # Initialize the GRU model
    model = GRUNet(
        input_size=config['input_size'],
        xx_size=config['xx_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size'],
        bias=config["bias"],
        output_type=config['output_type'],
        dropout=config['dropout'],
        seed=config['model_seed']
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adamax(model.parameters(), lr=config['learning_rate'])

    return model, criterion, optimizer

def evaluate_fn(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return -avg_loss

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer, 
          model_filename, eval_interval, evaluate_fn, logger, patients):
    """
    Train the GRU model.

    Parameters
    ----------
    model : nn.Module
        The GRU model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation dataset.
    criterion : nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    num_epochs : int
        Number of epochs to train the model.
    device : torch.device
        Device to run the model on ('cuda' or 'cpu').
    writer : SummaryWriter
        TensorBoard writer to log the training process.
    model_filename : string
        File name for saving the best model.
    eval_interval : int
        Evaluate model on validation set every `eval_interval` epochs.
    evaluate_fn : function
        A function to evaluate the model on validation set. Should return a scalar score to maximize.
        
    Notes
    -----
    This function performs the following during training:

    - Logs the training loss at each step to TensorBoard.
    - Logs the average training loss per epoch to TensorBoard.
    - Logs the gradients of all model parameters at each epoch to TensorBoard.
    - Every `eval_interval` epochs, evaluates the model on the validation set using a provided `evaluate_fn`.
    - The evaluation score is also logged to TensorBoard.
    - If the evaluation score is better than the previous best (i.e., higher is better), the current model is saved to the specified file.
      Note: If your evaluation metric is a loss (e.g., MSE), you should ensure that `evaluate_fn` returns the **negative** value of the loss,
      so that "higher is better" remains consistent.

    """
    global_step = 0
    best_score = float('-inf')

    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()

        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for the epoch
            epoch_loss += loss.item()

            # Log the loss for the current step
            global_step += 1
            writer.add_scalar(f'Loss/train', loss.item(), global_step)

        # Log the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar(f'Loss/epoch', avg_epoch_loss, epoch)

        # Print results for the epoch
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Time: {time.time() - start_time:.2f}s")

        # Log the gradients of parameters after the epoch
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}_epoch', param.grad, epoch)
        
        # Evaluate the current model
        if (epoch + 1) % eval_interval == 0 and evaluate_fn is not None:
            model.eval()
            with torch.no_grad():
                score = evaluate_fn(model, criterion, val_loader, device)
                writer.add_scalar('Score/val', score, epoch)

            # If current model is better, then save it
            if score > best_score:
                best_score = score
                counter = 0
                torch.save(model.state_dict(), model_filename)
                logger.info(f"New best model saved to {model_filename} (score: {score:.4f})")
            else:
                logger.debug(f"No improvement in validation score (current: {score:.4f}, best: {best_score:.4f})")
                counter += 1
            if counter >= patients:
                logger.info(f"Stopping training early after {counter} epochs without improvement.")
                break
            model.train()

def main():
    """
    Main function to train and evaluate the GRU model.
    """
    # Setup logger
    logger = setup_logger(config['log_file_name'])

    # Set device (GPU or CPU)
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    X_train, y_train, X_val, y_val, _, _ = load_data(config['data_file'], config['time_step'], config['test_ratio'],
            config['val_ratio'], config['split_seed'], logger, config['data_save_filename'], config['force_reload'])

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model, loss, optimizer
    model, criterion, optimizer = initialize_model(config, device)

    # TensorBoard writer
    writer = SummaryWriter(config['log_dir'])

    # Train the model with validation evaluation
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        writer=writer,
        model_filename=config['model_filename'],
        eval_interval=config['eval_interval'],
        patients=config['patients'],
        evaluate_fn=evaluate_fn,
        logger=logger
    )

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    # Configuration file for training the GRU model
    config = {
        # Data parameters
        'data_file': 'D:/桌面/毕设/2023国赛C/数据/大类/1.辣椒/classify_1_dataset_day.csv',            # Path to the CSV file containing input features and targets
        'time_step': 7,                     # Length of the sliding window (sequence length). Defines the number of time steps to consider for each input sequence.
        'val_ratio': 0.2,                   # Sample ratio for validation dataset
        'test_ratio': 0.2,                  # Sample ratio for test dataset
        'split_seed': 42,                   # Random seed to ensure reproducibility of data splits
        'force_reload': True,              # If True, the data will be reprocessed and resaved even if it has been divided before and the division results have been saved.

        # Model parameters
        'input_size': 11,                   # Number of input features (dimension of the input data)
        'xx_size': 11,                      # Number of linearly transformed features
        'num_layers': 3,                    # Number of stacked GRU layers
        'output_size': 1,                   # Number of output features (dimension of the final output)
        'bias': False,                      # Whether to use bias in GRU layers
        'output_type': 'mean',              # 'last' or 'mean', how to process GRU outputs
        'dropout': 0.1,                     # Dropout rate applied to GRU layers
        'model_seed':1234,                  # Random seed for initializing the GRUNet Model

        # Training parameters
        'learning_rate': 0.005,              # Learning rate for the optimizer
        'batch_size': 1078,                 # Batch size for training
        'num_epochs': 3000,                 # Number of epochs to train the model
        'eval_interval': 5,                 # Evaluate model on validation set every `eval_interval` epochs
        'patients':7,                       # Number of rounds of waiting for ealy stopping
        # TensorBoard parameters
        'log_dir': 'runs/gru_experiment',   # Directory to save TensorBoard logs
        'log_file_name': 'log1.log',

        # Saving parameters
        'model_filename': 'gru_model1.pth',  # Path to save the trained model
        'data_save_filename': 'data1.pt',    # path to save the divided dataset

        # Device configuration (GPU or CPU)
        'device': 'cuda',                   # Device to run the model on ('cuda' or 'cpu')
    }

    main()

