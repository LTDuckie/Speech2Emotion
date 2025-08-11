#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import import_ipynb
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ER_model import EmotionDataset, CNNEmotion

def train_and_evaluate(data_dir):
    # 获取所有音频文件路径
    files = glob.glob(f"{data_dir}/**/*.wav", recursive=True)
    # 从文件名中提取标签
    labels = [f.split("-")[2] for f in files]
    
    # 划分训练集和测试集
    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # 创建数据集和数据加载器
    train_dataset = EmotionDataset(train_files, train_labels)
    test_dataset = EmotionDataset(test_files, test_labels)  # 复用训练集的编码器
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    num_classes = len(train_dataset.encoder.classes_)
    model = CNNEmotion(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 跟踪最佳准确率和对应的模型参数
    best_acc = 0.0
    best_model_weights = None
    
    # 训练并在每个epoch后测试
    for epoch in range(40):
        # 训练阶段
        model.train()
        total_loss = 0
        for mel, label in train_loader:
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 测试阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for mel, label in test_loader:
                outputs = model(mel)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)
        
        current_acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {current_acc:.2%}")
        
        # 更新最佳模型
        if current_acc > best_acc:
            best_acc = current_acc
            best_model_weights = model.state_dict()
            print(f"New best accuracy: {best_acc:.2%}, saving model...")
    
    # 保存最佳模型和标签编码器
    torch.save(best_model_weights, "best_emotion_model.pth")
    torch.save(train_dataset.encoder.classes_, "classes.pt")
    print(f"Training complete. Best test accuracy: {best_acc:.2%}")
    print("Best model and label encoder saved.")

if __name__ == "__main__":
    train_and_evaluate("ravdess")


# In[ ]:




