
import torch

def cosine_metric(novels, bases):
    """
    Args:
        novel (tensor): [dim] -> [batch_size,dim]
        bases (tensro): [batch_size,dim]
    Returns:
    """    
    # batch_size = bases.shape[0] 
    # novels = novel.unsqueeze(0).expand(batch_size, -1) 

    n = novels.shape[0]  # [25]=[5-way,5-shot]
    m = bases.shape[1]  # [5]=[5-way,1-shot]
    novels = novels.unsqueeze(1).expand(n, m, -1)  # [25,5,1600]

    bases = bases.transpose(1,0)
    bases = bases.unsqueeze(0).expand(n, m, -1)  # [25,5,1600]

    assert novels.size() == bases.size()
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    sim_score = cos(bases, novels)

    # print(sim_score.shape)
    return sim_score

def euclidean_dis(a, b):
    n = a.shape[0]  # [25]=[5-way,5-shot]
    m = b.shape[0]  # [5]=[5-way,1-shot]
    a = a.unsqueeze(1).expand(n, m, -1)  # [25,5,1600]
    b = b.unsqueeze(0).expand(n, m, -1)  # [25,5,1600]
    # dis = torch.pow(a - b, 2).sum(dim=2).clamp(min=1e-12).sqrt()
    dis = torch.pow(a - b, 2).sum(dim=2).sqrt()
    return dis


def euclidean_metric(a, b):  # 欧式距离度量[query,support]
    """
    :param a:[query_shot*way,D]=[Q*5,1600]
    :param b:[way,D]=[5,1600]
    :return:[Q*N,way]=[50,5]
    """
    n = a.shape[0]  # [25]=[5-way,5-shot]
    m = b.shape[1]  # [5]=[5-way,1-shot]
    a = a.unsqueeze(1).expand(n, m, -1)  # [25,5,1600]

    b = b.transpose(1,0)
    b = b.unsqueeze(0).expand(n, m, -1)  # [25,5,1600]

    print(a.shape, b.shape)

    logits = -((a - b) ** 2).sum(dim=2)  # [Q*N,way]->[25,5]  query的25张图，每张图用一个5维向量表示标签信息,sum(dim=1600)
    # 这里距离用于计算acc和loss，所以不用开方也能比较大小.要计算softmax，而距离越大越不相似因此取反
    return logits  # [25,5]->25张query图像，分别与5张support的距离，取距离负数，取最大的下标，即为预测的类别