from lib2to3.pytree import LeafPattern
from operator import le
import numpy as np
import time


def loadData(fileName):
    '''
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    '''
    #存放数据及标记
    dataArr = []; labelArr = []
    #读取文件
    fr = open(fileName)
    #遍历文件中的每一行
    for line in fr.readlines():
        #获取当前行，并按“，”切割成字段放入列表中
        #strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
        #split：按照指定的字符将字符串切割成每个字段，返回列表形式
        curLine = line.strip().split(',')
        #将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
        #在放入的同时将原先字符串形式的数据转换为整型
        #此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        #将标记信息放入标记集中
        #放入的同时将标记转换为整型
        labelArr.append(int(curLine[0]))
    #返回数据集和标记
    return dataArr, labelArr


def getAllProbability(trainDataArr, trainLabelArr):
    '''
    通过训练集计算先验概率分布和条件概率分布
    :param trainDataArr: 训练数据集
    :param trainLabelArr: 训练标记集
    :return: 先验概率分布和条件概率分布
    '''
    #设置样本特征数目，数据集中手写图片为28*28，转换为向量是784维。
    featureNum = 784
    #设置类别数目，0-9共十个类别
    classNum = 10

    #Py = np.zeros(classNum)  1 x 10
    #初始化先验概率分布存放数组，后续计算得到的P(Y = 0)放在Py[0]中，以此类推
    #数据长度为10行1列 
    P_y = np.zeros((classNum, 1))
    #对每个类别进行一次循环，分别计算它们的先验概率分布
    for i in range(classNum):
        #np.mat(trainLabelArr) == i：将标签转换为矩阵形式，里面的每一位与i比较，若相等，该位变为Ture，反之False
        #np.sum(np.mat(trainLabelArr) == i):计算上一步得到的矩阵中Ture的个数，进行求和(直观上就是找所有label中有多少个
        #为i的标记，求得4.8式P（Y = Ck）中的分子)
        #np.sum(np.mat(trainLabelArr) == i)) + 1：参考“4.2.3节 贝叶斯估计”，例如若数据集总不存在y=1的标记，也就是说
        #手写数据集中没有1这张图，那么如果不加1，由于没有y=1，所以分子就会变成0，那么在最后求后验概率时这一项就变成了0，再
        #和条件概率乘，结果同样为0，不允许存在这种情况，所以分子加1，分母加上K（K为标签可取的值数量，这里有10个数，取值为10）
        #(len(trainLabelArr) + classNum)：标签集的总长度+classNum.   Laplace smoothing
        #((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10)：最后求得的先验概率
        P_y[i] = (np.sum(np.mat(trainDataArr) == i) + 1) / (len(trainLabelArr) + classNum)
    
    Py = np.log(P_y)

    #计算条件概率 Px_y=P（X=x|Y = y）
    #计算条件概率分成了两个步骤，下方第一个大for循环用于累加，参考书中“4.2.3 贝叶斯估计 式4.10”，下方第一个大for循环内部是
    #用于计算式4.10的分子，至于分子的+1以及分母的计算在下方第二个大For内
    #初始化为全0矩阵，用于存放所有情况下的条件概率
    Px_y = np.zeros((classNum, featureNum, 2))
    #对标记集进行遍历
    for i in range(len(trainLabelArr)):
        #获取当前循环所使用的标记
        label = trainLabelArr[i]
        #获取当前要处理的样本
        x = trainDataArr[i]
        #对该样本的每一维特诊进行遍历
        for j in range(featureNum):
            #在矩阵中对应位置加1
            #这里还没有计算条件概率，先把所有数累加，全加完以后，在后续步骤中再求对应的条件概率
            Px_y[label][j][x[j]] += 1
    
    
    #第二个大for，计算式4.10的分母，以及分子和分母之间的除法
    #循环每一个标记（共10个）
    for label in range(classNum):
        for j in range(featureNum):
            #获取y=label，第j个特征为0的个数
            Px_y0 = Px_y[label][j][0]
            #获取y=label，第j个特征为1的个数
            Px_y1 = Px_y[label][j][1]
            #对式4.10的分子和分母进行相除，再除之前依据贝叶斯估计，分母需要加上2（为每个特征可取值个数）
            #分别计算对于y= label，x第j个特征为0和1的条件概率分布
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    #返回先验概率分布和条件概率分布
    return Py, Px_y


def NaiveBayes(Py, Px_y, x):
    '''
    通过朴素贝叶斯进行概率估计
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param x: 要估计的样本x
    :return: 返回所有label的估计概率
    '''
    #设置特征数目
    featrueNum = 784
    #设置类别数目
    classNum = 10
    #建立存放所有标记的估计概率数组
    P = [0] * classNum

    #对于每一个类别，单独估计其概率
    for i in range(classNum):
        #初始化sum为0，sum为求和项。
        #在训练过程中对概率进行了log处理，所以这里原先应当是连乘所有概率，最后比较哪个概率最大
        #但是当使用log处理时，连乘变成了累加，所以使用sum
        sum = 0
        #获取每一个条件概率值，进行累加
        for j in range(featrueNum):
            sum += Px_y[i][j][x[j]]
        #最后再和先验概率相加
        P[i] = sum + Py[i]
    
    #max(P)：找到概率最大值
    #P.index(max(P))：找到该概率最大值对应的所有（索引值和标签值相等）
    return P.index(max(P))

def model_test(Py, Px_y, testDataArr, testLabelArr):
    '''
    对测试集进行测试
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param testDataArr: 测试集数据
    :param testLabelArr: 测试集标记
    :return: 准确率
    '''
    #错误值
    errorCnt = 0
    #循环遍历测试集中的每一个样本
    for i in range(len(testDataArr)):
        #获取预测值
        predict = NaiveBayes(Py, Px_y, testDataArr[i])
        #与答案进行比较
        if(predict != testLabelArr[i]): 
            #若错误  错误值计数加1
            errorCnt += 1
    
    #返回准确率
    return 1 - (errorCnt / len(testDataArr))

if __name__ == "__main__":
    start = time.time()
    # 获取训练集
    print('start read transSet')
    trainDataArr, trainLabelArr = loadData('D:\\WORKSPACE\\statistic-machine-learning\\Mnist\\mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataArr, testLabelArr = loadData('D:\\WORKSPACE\\statistic-machine-learning\\Mnist\\mnist_test.csv')

    #开始训练，学习先验概率分布和条件概率分布
    print('start to train')
    Py, Px_y = getAllProbability(trainDataArr, trainLabelArr)

    #使用习得的先验概率分布和条件概率分布对测试集进行测试
    print('start to test')
    accuracy = model_test(Py, Px_y, testDataArr, testLabelArr)

    #打印准确率
    print('the accuracy is:', accuracy)
    #打印时间
    print('time span:', time.time() -start)