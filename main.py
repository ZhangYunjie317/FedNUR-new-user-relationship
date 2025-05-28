import random
import numpy as np
import torch

from server import Server

# 设置随机种子
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Random seed set to {SEED}.")
print("Random number from Python:", random.randint(0, 100))
print("Random number from NumPy:", np.random.rand())
print("Random tensor from PyTorch:", torch.randn(1))

def main():
    # Step 0: 初始化服务器
    server = Server()
    # Step 1: 读取和拆分数据
    clients_data = server.load_and_distribute_data()
    # Step 2: 初始化客户端
    server.initialize_clients(clients_data)
    # Step 3: 模拟客户端本地训练
    server.simulate_local_training()




if __name__ == '__main__':
    main()
