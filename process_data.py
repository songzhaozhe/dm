import pandas as pd
import numpy as np
import os

def process_label(file):


def main():
    const_interval = 20
    needed_columns = ['LastPrice', 'LastVolume', 'LastTurnover', 'AskPrice1', 'BidPrice1', 'AskVolume1', 'BidVolume1','BidVolume2','BidVolume3']
    data_list = []
    path = "./data/m0000"
    files = os.listdir(path)
    array_list = []
    label_list = []
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            filenames.append(os.path.join(path, filename))
    filenames.sort()
        data_list.append(path+"/"+file)
        print (file_name)
        dF = pd.read_csv(file_name)
        array_list.append(dF.as_matrix(needed_columns))

    for day_matrix in array_list:
        row_num = day_matrix.shape[0]
        last_row_num = row_num#表示label只能到第几行，否则到底没有合适的label了
        dp = np.zeros(row_num)#往后推是涨了（1）还是跌了（0）。-1只会出现在末尾一串
        dp[row_num-1] = -1
        avg_price_next = (day_matrix[row_num-1,3] + day_matrix[row_num-1,4]) / 2
        for i in range(day_matrix.shape[0] - 2, -1, -1):
            avg_price_cur = (day_matrix[i,3] + day_matrix[i,4]) / 2
            if avg_price_cur < avg_price_next:
                dp[i] = 1
            elif avg_price_cur > avg_price_next:
                dp[i] = 0
            else:
                dp[i] = dp[i+1]
            avg_price_next = avg_price_cur

            j = i + const_interval
            if j >= row_num:
                day_matrix[i,-1] = -1
                break
            avg_price_interval = (day_matrix[j,3] + day_matrix[j,4]) / 2
            if avg_price_cur < avg_price_next:
                day_matrix[i,-1] = 1#1表示升了
            elif avg_price_cur > avg_price_next:
                day_matrix[i,-1] = 0
            else:
                if dp[j] == -1:

                day_matrix[i,-1] = dp[j]

#['LastPrice'#0, 'LastVolume'#1, 'LastTurnover'#2, 'AskPrice1'#3, 'BidPrice1'#4, 'AskVolume1'#5, 'BidVolume1'#6, #7, #8]
if __name__ == '__main__':
    main()
#pd.read_csv("")
#dataFrame.as_matrix(columns=None/[,])