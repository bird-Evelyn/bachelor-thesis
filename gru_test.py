import numpy as np
from gru import GRUNet
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data  import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import font_manager
#自定义RSE
def rse_scorer(true, pred):
    mse = mean_squared_error(true, pred)
    return mse / np.sum(np.square(true - np.mean(true)))

#自定义RAE
def rae_scorer(true, pred):
    true_mean = np.mean(true)
    squared_error_num = np.sum(np.abs(true - pred))
    squared_error_den = np.sum(np.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss


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
        'hidden_size': 16,                  # Number of features in the hidden state of the GRU
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#加载test_loader
data_dict = torch.load('D:/桌面/毕设/2023国赛C/model/gru/data1.pt')
#print(data_dict.keys())
X_test = data_dict['X_test']
y_test = data_dict['y_test']
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
#print(type(test_loader))

#模型导入
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

#加载模型参数
model.load_state_dict(torch.load('D:/桌面/毕设/2023国赛C/model/gru/gru_model1.pth'))
model.eval()

#定义损失函数
criterion ={
        'MAE': mean_absolute_error,
        'RAE': rae_scorer,
        'MSE': mean_squared_error,
        'RSE': rse_scorer,
        'MAPE': mean_absolute_percentage_error,
    }

all_preds = []
all_targets = []

#测试模型
total_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        #保存每一批次的输出和目标，
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
#然后再拼接成一个完整的，最后再计算
all_preds = np.concatenate(all_preds, axis=0).squeeze()
all_targets = np.concatenate(all_targets, axis=0).squeeze()

# 计算所有指标,criterion是个字典，这里要遍历每个字典的评价函数
results = {}
for name, func in criterion.items():
    results[name] = func(all_targets, all_preds)

# 打印所有指标
print("\n=== test Metrics ===")
for metric, testue in results.items():
    print(f"{metric}: {testue:.4f}")



# #可视化，但好像没啥用，因为这个看不出来啥😫
# font = font_manager.FontProperties(family='SimHei')
# def plot_true_vs_pred(all_targets,all_preds):
#     plt.figure(figsize=(8,6))
#     plt.plot(all_targets, label='True', marker='o')
#     plt.plot(all_preds, label='Predicted', marker='x')
#     plt.legend()
#     plt.xlabel('Sample')
#     plt.ylabel('testue')
#     plt.title('True vs Predicted_辣椒',fontproperties=font)
#     plt.grid()
#     plt.show()
#
# # 用法
# plot_true_vs_pred(all_targets, all_targets)
#
#



#不会的代码详解w(ﾟДﾟ)💪

##outputs.cpu().numpy()
###因为PyTorch的tensor如果是在GPU上，是不能直接转成numpy的，会报错。
###numpy只能处理 CPU 里的数组！
###所以 .cpu() 的意思是把这个tensor从GPU拷贝到CPU上。

##print(data_dict.keys())/print(type(test_loader))
###在加载数据的时候一定要看清数据的结构是什么样的

##make_scorer 是 sklearn.metrics.make_scorer
###本意是把一个普通的评价函数包装成交叉验证（如GridSearchCV）的评分函数。
###返回的是一个能在GridSearch里自动调用的对象，而不是普通函数！
###而我需要做的是直接计算指标，不需要交叉验证，所以用普通函数就ok