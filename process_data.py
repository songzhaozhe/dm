import pandas as pd
import numpy as np
import os
#['LastPrice'#0, 'LastVolume'#1, 'LastTurnover'#2, 'AskPrice1'#3, 'BidPrice1'#4, 'AskVolume1'#5, 'BidVolume1'#6, #7, #8]
def process_label(file_name, id, needed_columns):
    STEP = 20
    dF = pd.read_csv(file_name)
    day_matrix = dF.as_matrix(needed_columns)
    row_num = day_matrix.shape[0]
    last_row_num = row_num  # 表示label只能到第几行，否则搜到底也没有合适的label了
    dp = np.zeros(row_num)  # 往后推是涨了（1）还是跌了（0）。-1只会出现在末尾一串
    dp[row_num - 1] = -1
    for j in range(day_matrix.shape[0] - 2, -1, -1):
        avg_price_cur = (day_matrix[j, 3] + day_matrix[j, 4]) / 2
        avg_price_next = (day_matrix[j+1, 3] + day_matrix[j+1, 4]) / 2
        if avg_price_cur < avg_price_next:
            dp[j] = 1
        elif avg_price_cur > avg_price_next:
            dp[j] = 0
        else:
            dp[j] = dp[j + 1]

        i = j - STEP
        if i < 0:
            break
        avg_price_i = (day_matrix[i, 3] + day_matrix[i, 4]) / 2
        if avg_price_i < avg_price_cur:
            day_matrix[i, -1] = 1  # 1表示升了
        elif avg_price_i > avg_price_cur:
            day_matrix[i, -1] = 0
        else:
            if dp[j] == -1:
                last_row_num = i
            else:
                day_matrix[i, -1] = dp[j]
    day_matrix = day_matrix[0:last_row_num]


def main():
    const_interval = 20
    needed_columns = ['LastPrice', 'LastVolume', 'LastTurnover', 'AskPrice1', 'BidPrice1', 'AskVolume1', 'BidVolume1','BidVolume2','BidVolume3']
    path = "./data/m0000"
    files = os.listdir(path)
    for file_name in os.listdir(path):
        if file_name.endswith(".csv"):
            files.append(os.path.join(path, file_name))
    files.sort()

    for id in range(len(files)):
        process_label(files[id], id, needed_columns)

#['LastPrice'#0, 'LastVolume'#1, 'LastTurnover'#2, 'AskPrice1'#3, 'BidPrice1'#4, 'AskVolume1'#5, 'BidVolume1'#6, #7, #8]
if __name__ == '__main__':
    main()
#pd.read_csv("")
#dataFrame.as_matrix(columns=None/[,])