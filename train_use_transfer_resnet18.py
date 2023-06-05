
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from CatDataset import Cat
from ResNet import Flatten
from torchvision.models import resnet18


torch.manual_seed(1234)

epochs = 10
batch_size = 32
learn_rate = 1e-3

data_path = os.path.abspath('data')
file_name = 'train_list.txt'

train_data = Cat(data_path, 224, mode='train', filename=file_name)
val_data = Cat(data_path, 224, mode='val', filename=file_name)
test_data = Cat(data_path, 224, mode='test', filename=file_name)

train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
val_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0)


def evaluate(model, loader):

    '''
    验证模型准确率
    :param model: 模型
    :param loader: 验证数据
    :return acc: 准确率
    '''

    correct, total = 0, len(loader.dataset)

    for img, label in loader:

        with torch.no_grad():
            logits = model(img)
            predict = logits.argmax(dim=1)

        correct += torch.eq(predict, label).sum().float().item()

    acc = correct / total

    return acc


def main():

    trained_model = resnet18(pretrained=True)

    model = nn.Sequential(
        *list(trained_model.children())[:-1],
        Flatten(),
        nn.Linear(512, 12)
    )

    # 已存在模型参数
    if os.path.exists(os.path.join(os.path.abspath(''), 'cat_transfer.cptk')):
        model.load_state_dict(torch.load('cat_transfer.cptk'))

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    # 损失函数
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0.8742283950617284, 0

    for epoch in range(epochs):

        for step, (img, label) in enumerate(train_loader):

            logits = model(img)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:

                print('After {} steps, the loss is {}.'.format(step, loss.item()))

        acc = evaluate(model, val_loader)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'cat_transfer.cptk')

        print('After {} epochs, the accuracy is {}, the loss is {}'.format(epoch, best_acc, loss.item()))

    print('the best accuracy is {}.\nthe best epoch is {}.'.format(best_acc, best_epoch))

    model.load_state_dict(torch.load('cat_transfer.cptk'))
    test_acc = evaluate(model, test_loader)

    print('test accuracy is: ', test_acc)


if __name__ == '__main__':
    main()
