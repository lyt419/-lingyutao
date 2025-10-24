from numpy import*
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group=array([[1.0,1,1],[1.0,1.0],[0,0],[0,0.1]])
    labels=["A","A","B","B"]
    return group,labels

def file2matrix(filename):
    fr=open(filename)   #打开文件
    array_olines=fr.readlines() #从文件中读取一行
    number_lines=len(array_olines) 
    return_mat=zeros((number_lines,3))
    class_label_vector=[]
    index=0
    for line in array_olines:
        line=line.strip()
        list_from_line=line.split('\t')
        return_mat[index,:]=list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index+=1
    return return_mat,class_label_vector


file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
dating_data_mat, dating_labels = file2matrix(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

fig=plt.figure()
#图形中添加一个子图
#第一个数字 "1"：将图形划分为 1 行
#第二个数字 "1"：将图形划分为 1 列
#第三个数字 "1"：选择第 1 个（也是唯一的一个）子图位置
ax=fig.add_subplot(111)  

#scatter 绘制散点图
#选择第二列作为图的x轴，选择第三列作为y轴
ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2])
#第三个参数调整点的大小，第四个参数控制点的颜色
ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2],15.0*array(dating_labels),15.0*array(dating_labels))
ax.set_xlabel('第二特征')
ax.set_ylabel('第三特征')
ax.set_title('特征关系散点图')
#plt.show()


def autoNorm(dataSet):
    #数据进行归一化

    minVals = dataSet.min(0) # 计算每列的最小值
    maxVals = dataSet.max(0)  # 计算每列的最大值
    ranges = maxVals - minVals  # 计算每列数据的范围（最大值-最小值）
    normDataSet = zeros(shape(dataSet))    # 创建一个与原始数据集形状相同的零矩阵，用于存储归一化后的数据
    m = dataSet.shape[0]     # 获取数据集的行数
    normDataSet = dataSet - tile(minVals, (m, 1))   # 将原始数据减去最小值（使用tile函数将minVals复制m行，使其形状与dataSet匹配）
    normDataSet = normDataSet/tile(ranges, (m, 1))     # 将减去最小值后的数据除以范围，实现归一化到[0,1]范围
    return normDataSet, ranges, minVals   # 返回归一化后的数据集、每列的范围和每列的最小值

def datingClassTest():
    # 使用留出法：设置测试集比例（hold-out比例），这里使用50%的数据作为测试集
    hoRatio = 0.50  # 测试集占比（可调整，如0.3表示30%测试集）
    
    # 1. 加载数据并归一化
    datingDataMat, datingLabels = file2matrix(r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # 2. 划分训练集与测试集
    m = normMat.shape[0]  # 总样本数
    numTestVecs = int(m * hoRatio)  # 测试集样本数量
    errorCount = 0.0  # 错误预测次数

    # 3. 遍历测试集，逐一预测并统计错误率
    for i in range(numTestVecs):
        # 输入样本：测试集第i行；训练集：从numTestVecs行开始到最后（避免与测试集重复）
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], 
                                     datingLabels[numTestVecs:m], 3)  # k=3（可调整）
        
        # 打印预测结果与真实结果，便于观察
        print(f"预测类别: {classifierResult}, 真实类别: {datingLabels[i]}")
        
        # 若预测错误，错误次数+1
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    # 4. 计算并打印准确率
    errorRate = (errorCount / float(numTestVecs)) * 100  # 错误率（百分比）
    accuracy = 100 - errorRate  # 准确率（百分比）
    print(f"\n测试集总样本数: {numTestVecs}")
    print(f"错误预测次数: {int(errorCount)}")
    print(f"模型准确率: {accuracy:.2f}%")
#datingClassTest()

def classify0(inX, dataSet, labels, k):
   #KNN分类核心代码
    # 1. 计算输入样本与数据集所有样本的欧氏距离
    dataSetSize = dataSet.shape[0]  # 获取数据集行数（样本数量）
    # 复制inX为dataSetSize行，与数据集形状匹配，再计算差值
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # 差值平方
    sqDistances = sqDiffMat.sum(axis=1)  # 按行求和（得到每个样本的距离平方）
    distances = sqDistances ** 0.5  # 开平方，得到欧氏距离

    # 2. 对距离排序，获取距离最小的k个样本的索引
    sortedDistIndicies = distances.argsort()  # 对距离从小到大排序，返回索引

    # 3. 统计k个近邻中各类别的出现次数
    classCount = {}  # 字典：key=类别，value=该类别出现次数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 获取第i个近邻的类别
        # 若类别已在字典中，次数+1；否则初始化为1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 4. 对类别次数排序，返回出现次数最多的类别（即输入样本的预测类别）
    # 按字典value降序排序，operator.itemgetter(1)表示按值排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回次数最多的类别
def classify_person():
    """
    交互式输入三项特征，使用约会数据集做 KNN 分类，并输出印象结果。
    """
    # 定义类别对应的印象评价
    resultList = ["不喜欢", "一般喜欢", "非常喜欢"]
    
    # 1. 交互式获取用户输入（需输入数字，若输入非数字会报错，可后续优化异常处理）
    percentTats = float(input("请输入对方每周玩视频游戏的时间占比(%): "))
    ffMiles = float(input("请输入对方每年飞行的里程数(英里): "))
    iceCream = float(input("请输入对方每周消耗的冰淇淋量(升): "))
    
    # 2. 加载数据集并归一化
    datingDataMat, datingLabels = file2matrix(r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # 3. 准备输入样本（需与数据集特征顺序一致：飞行里程、游戏占比、冰淇淋量）
    inArr = array([ffMiles, percentTats, iceCream])
    
    # 4. 归一化输入样本（使用数据集的minVals和ranges，保证尺度一致）
    normInArr = (inArr - minVals) / ranges
    
    # 5. 调用KNN分类函数预测类别（k=3）
    classifierResult = classify0(normInArr, normMat, datingLabels, 3)
    
    # 6. 输出分类结果（classifierResult是1/2/3，对应resultList的索引0/1/2）
    print(f"\n根据分析，你对对方的印象可能是：{resultList[classifierResult - 1]}")

# —— 死循环调用 ——
if __name__ == "__main__":
    while True:
        # 提供用户选择菜单
        print("\n===== KNN约会分类模型 =====")
        print("1. 运行模型测试（计算准确率）")
        print("2. 运行交互式分类（输入特征预测印象）")
        print("3. 退出程序")
        choice = input("请输入你的选择(1/2/3): ")
        
        # 根据选择执行对应功能
        if choice == "1":
            print("\n—— 开始模型测试 ——")
            datingClassTest()
        elif choice == "2":
            print("\n—— 开始交互式分类 ——")
            classify_person()
        elif choice == "3":
            print("程序已退出，再见！")
            break  # 退出死循环
        else:
            print("输入错误，请重新输入1/2/3！")  # 处理无效输入
