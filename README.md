# corn_disease_detection
Detect plant disease with a leaf☘️image.

## Resnet
* 网络结构: 使用resnet移除最后一个全链接层还有一个conv网络,
用全局池化上提取的特征;拼接三层全链接层
```python
class DeeperFCClassify(nn.Module):
    def __init__(self, in_features, class_size):
        super(DeeperFCClassify, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features//2)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features//2, in_features//4)
        self.fc3 = nn.Linear(in_features//4, class_size)
    
    def forward(self, features):
        y = self.fc1(features)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y
```
* 训练结果: 训练100个epoch,准确率达到77%+
* 结论: 目前相对较好的model
* 其他: 使用resnet做的其他尝试:
  * 移除两个conv:训练结果停留在50~60%的准确率
  * 移除一个conv:使用一个全链接层,效果稍差70%左右;两个和三个全链接层,结果差不多,77%左右

## Densenet 
* 网络结构: 使用densenet移除最后一个全链接层,用全局池化上提取的特征;
拼接一层、两层全链接层
* 训练结果: 训练几十个epoch,没有看到聚合的趋势,准确率停留在10%左右
* 结论: bad model

