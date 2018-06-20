import csv
import numpy as np


def getDataSep(intput):
    reader = csv.reader(intput)
    out1 = []
    out2 = []
    out3 = []
    out4 = []
    ans = []
    ans_trans = {1: [1, 0, 0],
                 2: [0, 1, 0],
                 3: [0, 0, 1]}

    for row in reader:
        i = list(map(float, row[0].split(" ")))
        out1.append(i[0])
        out2.append(i[1])
        out3.append(i[2])
        out4.append(i[3])
        ans.append(ans_trans[i[4]])

    return [out1, out2, out3, out4], ans


def recognitionPerc(input, ans, nn):
    good_ans = 0
    length = len(input)
    for test in range(length):
        t = nn.query(input[test]).T
        good_ans += clas_test(ans[test], t[0])
    return good_ans / length * 100


def clas_test(ans, target):
    return (ans == netToAns(target)).all()


def netToAns(target):
    max = np.max(target)
    index_max = -1
    # find index_max
    for i in range(len(target)):
        if target[i] == max:
            index_max = i
    # Change index max to 1 and rest to 0
    for i in range(len(target)):
        if i == index_max:
            target[i] = 1
        else:
            target[i] = 0
    return target
