import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from p27_model import *

#准备数据集
train_data=torchvision.datasets.CIFAR10(root='./data',train=True,transform=torchvision.transforms.ToTensor(),
                                       download=True)
test_data=torchvision.datasets.CIFAR10(root='./data',train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)

#length长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为：{}".format(len(train_data)))
print("测试数据集的长度为：{}".format(len(test_data)))

#利用dataloader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)

#创建网络模型
tudui=Tudui()

#创建损失函数
loss_fn=nn.CrossEntropyLoss()

#优化器
learning_rate=0.01
optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step=0
#记录测试的次数
total_test_step=0
#训练的次数
epoch=10

#添加tensorboard
writer=SummaryWriter("./log_train")

for i in range(epoch):
    print("---------第{}轮训练开始--------".format(i+1))

    tudui.train()
    for data in train_dataloader:
        imgs,targets=data
        outputs=tudui(imgs)
        loss=loss_fn(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step+=1
        if total_train_step%100==0:
            print("训练次数：{},Loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss,total_train_step)

#在测试集上进行测试
    tudui.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
                imgs,targets=data
                outputs=tudui(imgs)
                loss=loss_fn(outputs,targets)
                total_test_loss+=loss
                accuracy=(outputs.argmax(1)==targets).sum()
                total_accuracy=total_accuracy+accuracy

    print("总体测试集上的loss:{}".format(total_test_loss))
    print("总体测试集上的准确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step+=1

    #存储模型每一轮的状态
    torch.save(tudui,"Tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()




