# documenting-cod
## usega
```python
  ...
#  判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')
        # 用来指定训练设备的参数,训练是在CPU上还是GPU上进行,如果args.is_cuda为True，则将设备设置为cuda:0，否则设置为cpu
 kwargs = {
            'num_workers': args.num_workers,
            # 指定数据加载器使用多少个线程来加载数据，提高数据加载的效率。args.num_workers 从命令行参数中获取的。其值决定了数据加载器在读取数据时使用的并行进程数量
            # 进程数量过高可能会导致系统资源占用过多，而影响其他进程的运行
            'pin_memory': True,
            # 是一个用来指定数据加载器的参数，指定数据加载器是否将数据加载到内存中
        } if args.is_cuda else {}

        # 载入数据
        train_dataset = datasets.ImageFolder(
            # 用来加载图像数据集的函数，args.train_dir是一个参数，是训练数据的目录路径，该路径会被传递给训练代码中的相关函数，使得数据可以被正确地读取和使用
            args.train_dir,
            # 帮助机器学习算法从图像中提取特征，从而更好地做出预测
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                # 是一种图像处理技术，进行随机大小裁剪，将图像的大小调整为256像素，在不改变图像的像素比例的情况下改变图像的大小
                # Transform是一个Python库，它提供了一组用于转换图像的工具，RandomRessizeCrop（256） 变换用于将图像随机裁剪为 256 像素大小
                transforms.ToTensor()
                # 将PIL图像或numpy.ndarray转换为张量（Tensor）类型。transforms.ToTensor()将图像像素的值从0-255转换为0-1的范围内的浮点数，并将其存储为张量
                # 张量是一种多维数组，它可以更有效地表示和处理数据
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
    ......
        self.train_loader = DataLoader(
            dataset=train_dataset,
            # 将定义数据集用于模型训练。train_dataset是一个变量，它包含了我们的训练数据,使用dataset=train_dataset来指定我们要使用的数据集
            batch_size=args.batch_size,
            # batch_size是指每一次模型训练时，输入的数据分成的小块的大小。这个值决定了一次训练中跑多少个样本
            shuffle=True,
            # shuffle=True在模型训练中的作用是使每个epoch中的训练数据顺序随机化，从而增加训练的随机性和稳定性,防止模型在顺序训练过程中出现输入相关的过拟合现象
            **kwargs
        )
        self.val_loader = DataLoader(
            # self.val_loader = DataLoader 是一个 Python 库，用于加载和组织机器学习模型的数据,有助于快速有效地将数据加载到模型中，并使其更易于处理和分析
            # 用于从各种源加载数据，例如 CSV 文件、数据库和其他源,有助于将数据组织成批次，从而更轻松地训练模型
            dataset=val_dataset,
            # val_dataset是以对特定目的有用的方式组织和格式化的数据集合
            batch_size=args.test_batch_size,
            # 在训练过程的每次迭代中使用的数据点数，因为它会影响模型的准确性以及训练模型所需的时间，较小的批量大小可以导致更快的训练，但模型的准确性可能较低
            shuffle=False,
            # 它不会打乱数据集中的数据，而是按照原来的顺序加载数据
            **kwargs
        )

        # 挑选神经网络、参数初始化
        ......
       
       self.model = net.to(self.device)
        # 将模型(net)移动到指定的设备(device)上进行训练，将神经网络从一个设备移动到另一个设备，例如计算机或移动设备，以便它可以用于不同的应用程序
        
        # 优化器
        self.optimizer = optim.AdamW(
            # AdamW是一种优化器，它是Adam优化器的变体，用于深度学习模型的训练，它的主要优点是它可以更有效地调整学习率
            self.model.parameters(),
            # 帮助我们识别模式，并且可以提供更准确的预测结果，帮助我们发现潜在的规律，从而改善我们的决策过程
            lr=args.lr
            # 是一个参数，可以用来控制机器学习算法的学习率，可以控制算法的收敛速度
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()
        # 定义模型训练过程中的损失函数，即交叉熵损失函数。它的作用是计算模型预测结果与目标值之间的差异，并根据这个差异来反向传播更新模型参数。
        # 交叉熵损失函数适用于多分类任务，因为它能够将模型预测的概率分布与真实概率分布之间的差异最小化。

        # warm up 学习率调整部分
        ......

            # 更新进度条
            .......

        # 打印验证结果
        ......
        
        # 返回重要信息，用于生成模型保存命名
        return 100. * num_correct / len(self.val_loader.dataset), validating_loss

if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)
    args = get_args()
    worker = Worker(args=args)
    # worker = Worker(args=args)的作用是创建一个Worker对象，在本地或远程计算机上进行多进程训练,并帮助在多个GPU、多个计算机上训练模型

    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        #  epoch 是一种用于编程的循环命令，它用于重复一组指令一定次数，如果要运行程序 10 次，则可以使用此命令执行此操作。它是自动化任务并确保程序多次正确运行的有用工具
        worker.train(epoch)
        val_acc, val_loss = worker.val()
        # val_acc是模型在验证集上的准确率，val_loss是模型在验证集上的损失值，这些指标可以用来衡量模型的性能，用于测量模型在验证集上的性能，以便确定模型是否可以在实际应用中使用
        .......












```
