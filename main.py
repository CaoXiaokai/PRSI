import numpy as np
import random
import string  
from dataloader import dataloader
from tqdm import tqdm
import time
import multiprocessing
from functools import partial
import os
import json
import logging

with open('./config.json', 'r') as f:
    config = json.load(f)

U = config['Number_user']
Number_group = config['Number_group']
Number_user_per_group = config['Number_user_per_group']
Secure_factor = config['Encryption_factor']
C = int(config['Number_splitting_vector']) #  row of encrypted matrix, the demension of the matrix is (C+1, M)
Len_virtualID = int(config['Len_virtualID']) # ID embedding
p_decay_rate = config['Attenuation_coefficient']
r = U // Number_group // Number_user_per_group
log = True if int(config['log']) == 1 else False

logger = None
if log:
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler("log.txt")
        ]
    )
    
    logger = logging.getLogger(__name__)

global server
global users
global num_group
global result_file
global I # = max_item_index + 1 # Total number of item
global M # =  Secure_factor * max_item_num # colomn of encrypted matrix

# dataloader
train_file = r'./dataset/train.txt'

def rank0_print(rank, content):
    if rank == 0:
        print(content)

def update_p(p: int):
    return p_decay_rate * p

def generate_random_string(length):
    # function for generating encrpted user ID which contain random letters and numbers
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class Server():
    def __init__(self) -> None:
        self.interact_matrix = np.zeros(shape=(U, I), dtype=int)        # recovered user interaction matrix
        self.encrypt_vector = dict()                                    # collected user splitting vector following the format of {user_VirtualID: encrypt_matrix (2, M)}
        self.user_map = dict()                                          # Maintain the row number of each user_VirtualID corresponding to the interaction matrix
        self.recommendation_result = np.zeros(shape=(U, I), dtype=int)  # recomendation result

    def Aggregate(self):
        '''
        Fucntions: aggregate the collected vectors from each user_VirtualID \
                and recover the original interaction matrix
        Inputs:
        Returns:

        '''
        cnt = 0
        for user_id_emb, encrypt_matrix in self.encrypt_vector.items():
            for i in range(M):
                if encrypt_matrix[1, i] == 1:
                    self.interact_matrix[cnt, encrypt_matrix[0, i]] = 1
            self.user_map[cnt] = user_id_emb # record the index of each line is from user_id_emb
            cnt += 1

        if cnt != U:
            raise ValueError("have not collected the data from all the users")

    def recommend(self):
        '''
        Fucntions: Server conduct recommendations algorithms and get recommendations results.
                    Here we do not care about what recommendations algorithm is used but the interaction 
                    prototol itself, so we do not implement a recommendation algorithm and jut
                    regard the interact_matrix as recommendations results
        Inputs:
        Returns:

        '''
        self.recommendation_result = self.interact_matrix.copy()

    def create_encrypted_vectors(self, data:np.array):
        """
        Function: a helper fucntion to spilt the recommendation result(vector) of one user (1, Num_item)
                 into C encrypted vecotrs with value of {-1, 0, 1} 
        Inputs: 
            data : (1, Num_item)
        Returns:
            encrypted_vector : (2, column)
        """
        non_zero_indices = list(np.nonzero(data)[0])
        interact_lst = [1 for _ in range(len(non_zero_indices))]
        st = set(non_zero_indices)
            
        while len(st) < M:
            confuse_item_id = np.random.randint(0, I)
            while confuse_item_id in st:
                confuse_item_id = np.random.randint(0, I)
            non_zero_indices.append(confuse_item_id)
            interact_lst.append(0)
            st.add(confuse_item_id)
            
        encrypted_vector = np.vstack((np.array(non_zero_indices), np.array(interact_lst))).T
        np.random.shuffle(encrypted_vector)
        encrypted_vector = encrypted_vector.T
        return encrypted_vector

    def vector_splitting(self):
        """
        Function: spilt the recommendation results(vectors) of each user into C encryption vectors
                 and send them to any client to start recommendation results sending phase.
                 Apply Alogoritm 1 in paper.
        Inputs: 
        Returns:
        """
        for i in range(U):
            vector_splitting_matrix = np.random.randint(-1, 2, size=(C+1, M))
            sum = np.sum(vector_splitting_matrix[1:,:], axis=0)
            encrypted_vector = self.create_encrypted_vectors(self.recommendation_result[i, :])

            diff =  encrypted_vector[1, :] - sum
            vector_splitting_matrix[0, :] = encrypted_vector[0, :]
            for k in range(len(diff)):
                vector_splitting_matrix[np.random.randint(1, vector_splitting_matrix.shape[0]), k] += diff[k]
 
            # random transfer the encrypted vectors to ant client
            belong_id_emb = self.user_map[i]
            pass_user_id = random.sample(range(0, U), C)
            for m, user_id in enumerate(pass_user_id):
                users[int(user_id)].recv.append((belong_id_emb, vector_splitting_matrix[[0, m+1], :]))


class User():
    def __init__(self, ip : int, len_virtualID: int, data: list) -> None:
        """
        inputs :
            ip: user_ip (for simplicity, we use user index for replacement)
            d_emd: the length of user encrypted ID string
            row: the number of client to transfer encrpted vector (the row of encrpted_matrix)
            column: number of column of encrpted matrix
            C: Number of vecotors to split 
            M: Total number of items(including real-interacted item and fake-iteracted items)
        """
        self.user_embedding = generate_random_string(length=len_virtualID) # virtual ID
        self.ip = ip                # IP address of user
        self.original_data = data   # original iteration vector(ony contain interation item ID)
        self.my_data = np.zeros(shape=(2, M), dtype=int)    # Encrypted iteration vector (containing real and fake-interacted items).
                                                            # The first row contains the item IDs, while the second row contains boolean values indicating 
                                                            # whether each item is a real interaction (True) or a fake interaction (False).
        self.compressed_vector = None                       # recovered recommendation vector
        self.encrypted_matrix = np.random.randint(-1, 2, size=(C+1, M)) # Matrix formed by C splitting vectors
        self.user_map = dict()  # 'LD' in paper used to store the (Virtial ID, IP) pairs for speeding up sending recommendation results phase
        self.recv = list()      # used to store received splitting vectors from other clients
        self.flag = True        # flag to indicate whether have to stop continuing random sending splitting vectors
        self.finish = False     # flag to indicate whether have finish the overall process
        self.num_receive = 0    # used to record the number of splitting vectors received
        self.p = 1.0            # R_{sto} in paper
        self.round = 0  
        self.recv_num = 0       # used to determine whether have receive all the encrypted vector
        self.forward_send = 0   # used to count the number of vector have sent in collecting iteration vectors phase
        self.backward_send = 0  # used to count the number of vector have sent in sending recommendation results phase

    def create_encrypted_vector(self):
        """
        Function: a helper fucntion to use fake iteration items to protect the real iteration items
                and spilt the processed iteraction vector into C encrypted vecotrs with value of {-1, 0, 1}.
        Inputs: 
            original_data : (1, Num_item it have)
        Returns:
            encrypted_vector : with shape (2, column) with the first row contains the item IDs, 
                                while the second row contains boolean values indicating 
        """
        non_zero_indices = self.original_data.copy()
        interact_lst = [1 for _ in range(len(non_zero_indices))]
        st = set(non_zero_indices)
        # 1. use fake iteration items to protect the real iteration items
        while len(st) < M:
            confuse_item_id = np.random.randint(0, I)
            while confuse_item_id in st:
                confuse_item_id = np.random.randint(0, I)
            non_zero_indices.append(confuse_item_id)
            interact_lst.append(0)
            st.add(confuse_item_id)
        
        self.compressed_vector = np.vstack((np.array(non_zero_indices), np.array(interact_lst))).T
        np.random.shuffle(self.compressed_vector)
        self.compressed_vector = self.compressed_vector.T
            
    def vector_splitting(self):
        """
        Function: spilt the processed iteraction vector (2, M) into C encrypted vecotrs with value of {-1, 0, 1}.
                Apply Alogoritm 1 in paper.
        Inputs: 
            processed iteraction vector : with shape (2, M)
        Returns: 
            encryption matrix: with shape (C+1, M)
        """
        self.encrypted_matrix[0, :] = self.compressed_vector[0, :]
        sum = np.sum(self.encrypted_matrix[1:,:], axis=0)
        diff = self.compressed_vector[1, :] - sum
        # get the encrypt matrix
        for i in range(len(diff)):
            self.encrypted_matrix[np.random.randint(1, self.encrypted_matrix.shape[0]), i] += diff[i]
        
        for i in range(1, self.encrypted_matrix.shape[0]):
            self.recv.append((self.user_embedding, self.encrypted_matrix[[0, i], :]))

    def pass_to_server(self):
        """
        Function: pass all the splitting vectors it currently have to the server
        """
        for (user_emb, encrpted_vector) in self.recv:
            if user_emb in server.encrypt_vector.keys():
                server.encrypt_vector[user_emb][1, :] += encrpted_vector[1, :]
            else:
                server.encrypt_vector[user_emb] = encrpted_vector

    def updata_user_map(self, user_ip: int):
        """
        Function: update the 'user_map'(LD) of the user with the given IP address.
        Inputs:
            user_ip : the IP address of the user.
        """
        if self.user_embedding in users[user_ip].user_map:
            return 
        else:
            users[user_ip].user_map[self.user_embedding] = self.ip

    def forward_pass_to_user(self):
        """
        Function: Apply Algorithm 2 in paper. 
        """
        prob = random.random()
        if self.flag:
            self.round += 1
            self.forward_send += len(self.recv)
            if prob < self.p:
                # pass to other users
                cnt = 0
                while cnt < len(self.recv):
                    pass_user_id = np.random.randint(0, U)
                    while pass_user_id == self.ip:
                        pass_user_id = np.random.randint(0, U)
                    users[pass_user_id].recv.append(self.recv[cnt])
                    self.updata_user_map(pass_user_id)
                    cnt += 1
            else:
                # pass encrypted vector to server
                self.pass_to_server()
                self.flag = False
            # update the probability to continue to pass the encrypted vector received to other users
            self.p = update_p(self.p)
        else:
            self.pass_to_server()
        # clear the recieve list
        self.recv = list()

    def backward_pass_to_user(self):
        """
        Function: Apply Algorithm 3 in paper.
        """
        for user_id_emb, encrpted_vector in self.recv:
            # find its own data
            if user_id_emb == self.user_embedding:
                
                self.my_data[0, :] = encrpted_vector[0, :]
                if not (encrpted_vector[0, :] == self.my_data[0, :]).all():
                    raise ValueError(f"User {self.ip} received inconsistent recovered vector")
                self.my_data[1, :] += encrpted_vector[1, :] 
                self.num_receive += 1
                if self.num_receive == C:
                    self.finish = True
            # accroding to the user map to find the user to pass to
            else:
                self.backward_send += 1
                # there is record of the current user_id_emb
                if user_id_emb in self.user_map.keys():
                    users[int(self.user_map[user_id_emb])].recv.append((user_id_emb, encrpted_vector))
                
                # if not found 
                else:
                    random_user_ip = np.random.randint(0, U)
                    users[random_user_ip].recv.append((user_id_emb, encrpted_vector))
            
        # clear the recv
        self.recv = list()

    def init(self):
        """
        Function: intialization to create slitting vectors
        """
        # first create encrpted vector(compressed vector)
        # (1, MAX_ITEM) -> (2, M)
        self.create_encrypted_vector()
        # fractorize the encrypted vector
        # (2, M) -> (1+C, M)
        self.vector_splitting()

    def get_recommend_result(self) -> int:
        """
        Function: get recommended results and to verify teh correctness of the algorithm
        """
        non_zero_index = np.nonzero(self.my_data[1])[0]
        recommend_result_item = np.sort(self.my_data[0][non_zero_index])

        if np.array_equal(recommend_result_item, np.sort(self.original_data)):
            return 1
        else:
            return 0

def run(rank, U, train_data, n):
    global users, server
    # instance
    rank0_print(rank, "-> create users and server")
    users = [User(ip=i, len_virtualID=Len_virtualID, data=train_data[i + rank * Number_user_per_group + (U // r) * n]) for i in range(U)]
    server = Server()

    start = time.time()
    rank0_print(rank, "-> start initializing...")
    # initialization
    for i in range(U):
        users[i].init()
    
    # forward process
    # logger.info("-> start forward process\n")
    rank0_print(rank, "-> start forward process")
    stop_user = set()
    while len(stop_user) < U:
        for i in range(U):
            if users[i].flag == False:
                stop_user.add(users[i].user_embedding)
            users[i].forward_pass_to_user()
    
    for i in range(U):
        users[i].pass_to_server()

    end1 = time.time()
    # logger.info(f"forward process time elapse: {int(end1 - start) // 60}min{int(end1 - start) % 60}sec\n")

    # server processing
    server.Aggregate()
    server.recommend()
    server.vector_splitting()

    finish_user = set()
    # backward
    # log.write("-> start backward process\n")
    rank0_print(rank, "-> start backward process")
    while len(finish_user) < U:
        for i in range(U):
            if users[i].finish == True:
                finish_user.add(users[i].user_embedding)
            users[i].backward_pass_to_user()

    end2 = time.time()
    # logger.info(f"Total time elapse: {int(end2 - start) // 60}min{int(end2 - start) % 60}sec\n")

    # eval
    num_correct_user = 0
    for i in range(U):
        num_correct_user += users[i].get_recommend_result()
    
    # record forward and backward communication cost
    if log: 
        forward_com_cost = 0
        for i in range(len(users)):
            forward_com_cost += users[i].forward_send
        logger.info(f"Group {rank} forward communication cost: {forward_com_cost}\n")
        backward_com_cost = 0
        for i in range(len(users)):
            backward_com_cost += users[i].backward_send
        logger.info(f"Group {rank} backward communication cost: {backward_com_cost}\n")
        logger.info(f'Group {rank} -> The total number of users join sercure RS: {U}\nThe total number of user recover correctly: {num_correct_user}\nSuccess rate: {100 * num_correct_user / U}%\n\n')

if __name__ == "__main__":

    print(f"Round: {r}, parallal_group: {Number_group}, User_each_group: {Number_user_per_group}")
    
    train_data, max_item_num, max_item_index = dataloader(train_file=train_file,
                                                    num_user=U)

    I = max_item_index + 1 # Total number of item
    M =  Secure_factor * max_item_num # colomn of encrypted matrix

    print(f"max_item_num: {max_item_num}, max_item_index: {max_item_index}")

   
    logger.info(f"Total user: {U}, Number_group: {Number_group}, Group_user: {Number_user_per_group}, max_item_num: {max_item_num}, max_item_index: {max_item_index}\n\n")

    # parallel 
    for i in range(r):
        num_processes = min(Number_group, multiprocessing.cpu_count()) # Number of processes, limited to number of groups or CPU count
        pool = multiprocessing.Pool(processes=num_processes)
        func = partial(run, U=U, train_data=train_data, n=i)
        pool.map(func, range(Number_group)) 
        pool.close()
        pool.join()

    print("finish")
            
