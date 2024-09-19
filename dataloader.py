import os 
import numpy as np

def dataloader(train_file = r'dataset/train.txt', num_user : int = 100):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    traindataSize = 0
    max_item_index = 0
    n_user = 0
    cnt = 0
    max_num_Item = 0
    with open(train_file) as f:
        for l in f.readlines():
            if cnt >= num_user:
                break
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                # uid = int(l[0])
                trainUniqueUsers.append(cnt)
                trainItem.append(items)
                max_item_index = max(max_item_index, max(items))
                max_num_Item = max(max_num_Item, len(items))
                n_user = max(n_user, cnt)
                traindataSize += len(items)
            cnt += 1

        return trainItem, max_num_Item, max_item_index

if __name__ == "__main__":
    # max_num_Item 指的是所有用户中交互过的最多的商品数
    trainItem, max_num_Item, max_item_index = dataloader(num_user=500)
    print(len(trainItem), max_num_Item, max_item_index)
    print(type(trainItem))