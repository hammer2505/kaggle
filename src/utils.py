import pandas as pd
import numpy as np
INTERMEDIADATAFOLD = '../intermediadata/'


def saveDataFrametoCSV(m_dataframe, filename):
    filename = INTERMEDIADATAFOLD + filename + '.csv'
    m_dataframe.to_csv(filename, index=False, header=False)


def readDataFramefromCSV(filename):
    filename = INTERMEDIADATAFOLD + filename + '.csv'
    return pd.read_csv(filename, header=None)


def saveListtoCSV(m_list, filename):
    for i in range(len(m_list)):
        saveDataFrametoCSV(m_list[i], filename+'_%d' % i)


def readListfromCSV(list_length, filename):
    m_list = [0] * list_length
    for i in range(list_length):
        m_list[i] = readDataFramefromCSV(filename+'_%d' % i)
    return m_list[:]

#a = pd.DataFrame(np.random.randn(4,5))
#saveDataFrametoCSV(a,'a')
#b = readDataFramefromCSV('a')
#print a.sub(b)
#b = b.T
#b.reset_index(drop=True,inplace=True)
#b = b.T
#print a.sub(b)
