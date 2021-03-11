import torch
import numpy as np
from torch.nn import functional
from tqdm import tqdm


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    qB = torch.sign(qB)
    rB = torch.sign(rB)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_precisions_topn(qB, rB, query_L, retrieval_L, recall_gas=0.02, num_retrieval=10000):
    qB = torch.from_numpy(qB.astype(np.float))
    rB = torch.from_numpy(rB.astype(np.float))
    qB = torch.sign(qB - 0.5)
    rB = torch.sign(rB - 0.5)
    num_query = query_L.shape[0]
    # num_retrieval = retrieval_L.shape[0]
    precisions = [0] * int(1 / recall_gas)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        for i, recall in enumerate(np.arange(recall_gas, 1 + recall_gas, recall_gas)):
            total = int(num_retrieval * recall)
            right = torch.nonzero(gnd[: total]).squeeze().numpy()
            # right_num = torch.nonzero(gnd[: total]).squeeze().shape[0]
            right_num = right.size
            precisions[i] += (right_num/total)
    for i in range(len(precisions)):
        precisions[i] /= num_query
    return precisions


def calc_precisions_hash(qB, rB, query_L, retrieval_L):
    qB = qB.float()
    rB = rB.float()
    qB = torch.sign(qB - 0.5)
    rB = torch.sign(rB - 0.5)
    num_query = query_L.shape[0]
    num_retrieval = retrieval_L.shape[0]
    bit = qB.shape[1]
    hamm = calc_hammingDist(qB, rB)
    hamm = hamm.type(torch.ByteTensor)
    total_num = [0] * (bit + 1)
    max_hamm = int(torch.max(hamm))
    gnd = (query_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze()
    total_right = torch.sum(torch.matmul(query_L, retrieval_L.t())>0)
    precisions = np.zeros([max_hamm + 1])
    recalls = np.zeros([max_hamm + 1])
    # _, index = torch.sort(hamm)
    # del _
    # for i in range(index.shape[0]):
    #     gnd[i, :] = gnd[i, index[i]]
    # del index
    right_num = 0
    recall_num = 0
    for i, radius in enumerate(range(0, max_hamm+1)):
        recall = torch.nonzero(hamm == radius)
        right = gnd[recall.split(1, dim=1)]
        recall_num += recall.shape[0]
        del recall
        right_num += torch.nonzero(right).shape[0]
        del right
        precisions[i] += (right_num / (recall_num + 1e-8))
        # recalls[i] += (recall_num / num_retrieval / num_query)
        recalls[i] += (recall_num / total_right)
    return precisions, recalls


def calc_precisions_hamming_radius(qB, rB, query_L, retrieval_L, hamming_gas=1):
    num_query = query_L.shape[0]
    bit = qB.shape[1]
    precisions = [0] * int(bit / hamming_gas)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        for i, recall in enumerate(np.arange(1, bit+1, hamming_gas)):
            total = torch.nonzero(hamm <= recall).squeeze().shape[0]
            if total == 0:
                precisions[i] += 0
                continue
            right = torch.nonzero(gnd[: total]).squeeze().numpy()
            right_num = right.size

            precisions[i] += (right_num / total)
    for i in range(len(precisions)):
        precisions[i] /= num_query
    return precisions

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf1(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算

def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = label1.matmul(label2.transpose(0, 1)) > 0
    return Sim.float()


def calc_neighbor_n(label1, label2,hash1,hash2):
    # calculate the similar matrix
    sim2=label1.matmul(label2.transpose(0, 1))
    sim1=hash1.matmul(hash2.transpose(0, 1))
    sim3=sim2-sim1
    sim_final=sim1.dot(np.exp(sim3))>0
    return sim_final.float()

def calc_neighbor_new(label1, label2,hash1,hash2):
    # calculate the similar matrix
    sim2 = label1.matmul(label2.transpose(0, 1))
    # sim2=functional.cosine_similarity(label1,label2.transpose(0, 1))
    sim1=functional.cosine_similarity(hash1,hash2)
    sim3=sim2-sim1
    sim_final=sim1.dot(np.exp(sim3))
    return sim_final.float()

def calc_agreement(hash1,hash2):
    agree=functional.cosine_similarity(hash1,hash2)
    return agree.float()


def norm_max_min(x: torch.Tensor, dim=None):
    if dim is None:
        max = torch.max(x)
        min = torch.min(x)
    if dim is not None:
        max = torch.max(x, dim=dim)[0]
        min = torch.min(x, dim=dim)[0]
        if dim > 0:
            max = max.unsqueeze(len(x.shape) - 1)
            min = min.unsqueeze(len(x.shape) - 1)
    norm = (x - min) / (max - min)
    return norm


def norm_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = (x - mean) / std
    return norm


def norm_abs_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = torch.abs(x - mean) / std
    return norm


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def calc_IF(all_bow):
    word_num = torch.sum(all_bow, dim=0)
    total_num = torch.sum(word_num)
    IF = word_num / total_num
    return IF


# def calc_loss(B, F, G, Sim, gamma1, gamma2, eta):
#     theta = torch.matmul(F, G.transpose(0, 1)) / 2
#     inter_loss = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
#     theta_f = torch.matmul(F, F.transpose(0, 1)) / 2
#     intra_img = torch.sum(torch.log(1 + torch.exp(theta_f)) - Sim * theta_f)
#     theta_g = torch.matmul(G, G.transpose(0, 1)) / 2
#     intra_txt = torch.sum(torch.log(1 + torch.exp(theta_g)) - Sim * theta_g)
#     intra_loss = gamma1 * intra_img + gamma2 * intra_txt
#     quan_loss = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2)) * eta
#     # term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
#     # loss = term1 + gamma * term2 + eta * term3
#     loss = inter_loss + intra_loss + quan_loss
#     return loss


# if __name__ == '__main__':
#     qB = torch.Tensor([[1, -1, 1, 1],
#                        [-1, -1, -1, 1],
#                        [1, 1, -1, 1],
#                        [1, 1, 1, -1]])
#     rB = torch.Tensor([[1, -1, 1, -1],
#                        [-1, -1, 1, -1],
#                        [-1, -1, 1, -1],
#                        [1, 1, -1, -1],
#                        [-1, 1, -1, -1],
#                        [1, 1, -1, 1]])
#     query_L = torch.Tensor([[0, 1, 0, 0],
#                             [1, 1, 0, 0],
#                             [1, 0, 0, 1],
#                             [0, 1, 0, 1]])
#     retrieval_L = torch.Tensor([[1, 0, 0, 1],
#                                 [1, 1, 0, 0],
#                                 [0, 1, 1, 0],
#                                 [0, 0, 1, 0],
#                                 [1, 0, 0, 0],
#                                 [0, 0, 1, 0]])
#
#     map = calc_map_k(qB, rB, query_L, retrieval_L)
#     print(map)
