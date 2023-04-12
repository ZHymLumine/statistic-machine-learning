'''
数据集：Mnist
训练集数量：60000(实际使用：10000)
测试集数量：10000（实际使用：1000)
层数：40
------------------------------
运行结果：
    正确率：97%
    运行时长：65m
'''

import time
import numpy as np

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

        #转换成二分类任务
        #标签0设置为1，反之为-1
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    #返回数据集和标记
    return dataArr, labelArr


def calc_e_Gx(trainDataArr, trainLabelArr, n, div, rule, D):
    '''
    计算分类错误率
    :param trainDataArr:训练数据集数字
    :param trainLabelArr: 训练标签集数组
    :param n: 要操作的特征
    :param div:划分点(特征取值)
    :param rule:正反例标签
    :param D:权值分布D
    :return:预测结果， 分类误差率
    '''
    #初始化分类误差率为0
    e = 0
    #将训练数据矩阵中特征为n的那一列单独剥出来做成数组。因为其他元素我们并不需要，
    #直接对庞大的训练集进行操作的话会很慢
    x = trainDataArr[:, n]
    #同样将标签也转换成数组格式，x和y的转换只是单纯为了提高运行速度
    #测试过相对直接操作而言性能提升很大
    y = trainLabelArr
    predict = []

    if rule == 'LisOne': L = 1; H = -1
    else :               L = -1; H = 1

    #遍历所有样本的特征
    for i in range(trainDataArr.shape[0]):
        if x[i] < div:
            #如果小于划分点，则预测为L
            #如果设置小于div为1，那么L就是1，
            #如果设置小于div为-1，L就是-1
            predict.append(L)
            #如果预测错误，分类错误率要加上该分错的样本的权值（8.1式）
            if y[i] != L: e += D[i]
        elif x[i] >= div:
            #与上面思想一样
            predict.append(H)
            if y[i] != H: e += D[i]
    #返回预测结果和分类错误率e
    #预测结果其实是为了后面做准备的，在算法8.1第四步式8.4中exp内部有个Gx，要用在那个地方
    #以此来更新新的D    
    return np.array(predict), e


def createSingleBoostingTree(trainDataArr, trainLabelArr, D):
    '''
    创建单层提升树
    :param trainDataArr:训练数据集数组
    :param trainLabelArr: 训练标签集数组
    :param D: 算法8.1中的D
    :return: 创建的单层提升树
    '''
    #获得样本数目及特征数量
    m, n = np.shape(trainDataArr)
    #单层树的字典，用于存放当前层提升树的参数
    #也可以认为该字典代表了一层提升树
    singleBoostTree = {}
    #初始化分类误差率，分类误差率在算法8.1步骤（2）（b）有提到
    #误差率最高也只能100%，因此初始化为1
    singleBoostTree['e'] = 1


    #对每一个特征进行遍历，寻找用于划分的最合适的特征
    for i in range(n):
        #因为特征已经经过二值化，只能为0和1，因此分切分时分为-0.5， 0.5， 1.5三挡进行切割
        for div in [-0.5, 0.5, 1.0]:
            #在单个特征内对正反例进行划分时，有两种情况：
            #可能是小于某值的为1，大于某值得为-1，也可能小于某值得是-1，反之为1
            #因此在寻找最佳提升树的同时对于两种情况也需要遍历运行
            #LisOne：Low is one：小于某值得是1
            #HisOne：High is one：大于某值得是1
            for rule in ['LisOne', 'HisOne']:
                #按照第i个特征，以值div进行切割，进行当前设置得到的预测和分类错误率
                Gx, e = calc_e_Gx(trainDataArr, trainLabelArr, i, div, rule, D)
                #如果分类错误率e小于当前最小的e，那么将它作为最小的分类错误率保存
                if e < singleBoostTree['e']:
                    singleBoostTree['e'] = e
                    #同时也需要存储最优划分点、划分规则、预测结果、特征索引
                    #以便进行D更新和后续预测使用
                    singleBoostTree['div'] = div
                    singleBoostTree['rule'] = rule
                    singleBoostTree['Gx'] = Gx
                    singleBoostTree['feature'] = i
    #返回单层的提升树
    return singleBoostTree    


def createBoostingTree(trainDataList, trainLabelList, treeNum = 50):
    '''
    创建提升树
    创建算法依据“8.1.2 AdaBoost算法” 算法8.1
    :param trainDataList:训练数据集
    :param trainLabelList: 训练测试集
    :param treeNum: 树的层数
    :return: 提升树
    '''
    #将数据和标签转化为数组形式
    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trainLabelList)
    #每增加一层数后，当前最终预测结果列表
    finallpredict = [0] * len(trainLabelArr)
    #获得训练集数量以及特征个数
    m, n = np.shape(trainDataArr)

    #依据算法8.1步骤（1）初始化D为1/N
    D = [1 / m] * m
    #初始化提升树列表，每个位置为一层
    tree = []
    #循环创建提升树

    for i in range(treeNum):
        curTree = createSingleBoostingTree(trainDataArr, trainLabelArr, D)
        alpha = 1/2 * np.log((1 - curTree['e']) / curTree['e'])
        Gx = curTree['Gx']
        #np.multiply(trainLabelArr, Gx)：exp中的y*Gm(x)，结果是一个行向量，内部为yi*Gm(xi)
        #np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))：上面求出来的行向量内部全体
        #成员再乘以-αm，然后取对数，和书上式子一样，只不过书上式子内是一个数，这里是一个向量
        #D是一个行向量，取代了式中的wmi，然后D求和为Zm
        #书中的式子最后得出来一个数w，所有数w组合形成新的D
        D =  np.multiply(D, np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))) / sum(D)
        #在当前层参数中增加alpha参数，预测的时候需要用到
        curTree['alpha'] = alpha
        #将当前层添加到提升树索引中。
        tree.append(curTree)

        #根据8.6式将结果加上当前层乘以α，得到目前的最终输出预测
        finallpredict += alpha * Gx
        #计算当前最终预测输出与实际标签之间的误差
        error = sum([1 for i in range(len(trainDataList)) if np.sign(finallpredict[i]) != trainLabelArr[i]])
        #计算当前最终误差率
        finallError = error / len(trainDataList)
        #如果误差为0，提前退出即可，因为没有必要再计算算了
        if finallError == 0:    return tree
        #打印一些信息
        print('iter:%d:%d, sigle error:%.4f, finall error:%.4f'%(i, treeNum, curTree['e'], finallError ))
    #返回整个提升树
    return tree


def predict(x, div, rule, feature):
    '''
    输出单独层预测结果
    :param x: 预测样本
    :param div: 划分点
    :param rule: 划分规则
    :param feature: 进行操作的特征
    :return:
    '''
    #依据划分规则定义小于及大于划分点的标签
    if rule == 'LisOne':    L = 1; H = -1
    else:                   L = -1; H = 1

    #判断预测结果
    if x[feature] < div: return L
    else:       return H


def model_test(testDataList, testLabelList, tree):
    '''
    测试
    :param testDataList:测试数据集
    :param testLabelList: 测试标签集
    :param tree: 提升树
    :return: 准确率
    '''
    #错误率计数值
    errorCnt = 0
    #遍历每一个测试样本
    for i in range(len(testDataList)):
        result = 0
        #依据算法8.1式8.6
        #预测式子是一个求和式，对于每一层的结果都要进行一次累加
        #遍历每层的树
        for curTree in tree:
            #获取该层参数
            div = curTree['div']
            rule = curTree['rule']
            feature = curTree['feature']
            alpha = curTree['alpha']
            #将当前层结果加入预测中
            result += alpha * predict(testDataList[i], div, rule, feature)
        #预测结果取sign值，如果大于0 sign为1，反之为0
        if np.sign(result) != testLabelList[i]: errorCnt += 1
    
    return 1 - errorCnt / len(testLabelList)


if __name__ == '__main__':
    #开始时间
    start = time.time()

    # 获取训练集
    print('start read transSet')
    trainDataList, trainLabelList = loadData('D:\\WORKSPACE\\statistic-machine-learning\\Mnist\\mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataList, testLabelList = loadData('D:\\WORKSPACE\\statistic-machine-learning\\Mnist\\mnist_test.csv')

    #创建提升树
    print('start init train')
    tree = createBoostingTree(trainDataList[:10000], trainLabelList[:10000], 40)

    #测试
    print('start to test')
    accuracy = model_test(testDataList[:1000], testLabelList[:1000], tree)
    print('the accuracy is:%d' % (accuracy * 100), '%')

    #结束时间
    end = time.time()
    print('time span:', end - start)