from mlagents.envs.environment import UnityEnvironment
import numpy as np

def preprocess_observation(obs):
    """
    for each agent in unity env, #为每一个agent找到与他最近的一个agent，
    和与两个agent最近的n个food和n个 badfood 表示为（1+1+n+n）x（1+1+n+n）
    维度的邻接矩阵 A，和（1+1+n+n）x l的特征矩阵 X
    return A, X
    """
    agentInfos = []
    foodInfos = []
    badFoodsInfos = []
    obs_ = []
    for i in range(int(len(obs)/6)):
        obs_.append(obs[i * 6:(i + 1) * 6])
        if i < 5:
            agentInfos.append(obs[i*6:(i+1)*6])
        elif i < 55:
            foodInfos.append(obs[i * 6:(i + 1) * 6])
        elif i < 105:
            badFoodsInfos.append(obs[i * 6:(i + 1) * 6])
    # for o in obs_:
    #     print(o)



    A_agent = []
    # dis = []

    for j in range(len(agentInfos)):
        f = []
        for r in range(len(agentInfos)):
            f.append([(agentInfos[r][0] - agentInfos[j][0]) ** 2 + (agentInfos[r][1] - agentInfos[j][1]) ** 2, r])
        f.sort(key=lambda x: x[0])
        A_agent.append([f[0][1], f[1][1]])
    # print(A_agent) #[[0, 2], [1, 2], [2, 1], [3, 4], [4, 1]]


    n = 4
    A_agent_food = []
    for a in A_agent:
        f = []
        for r in range(len(foodInfos)):
            f.append([
                (foodInfos[r][0] - agentInfos[a[0]][0]) ** 2 + (foodInfos[r][1] - agentInfos[a[0]][1]) ** 2 +
                (foodInfos[r][0] - agentInfos[a[1]][0]) ** 2 + (foodInfos[r][1] - agentInfos[a[1]][1]) ** 2
                , r
            ])
        f.sort(key=lambda x: x[0])
        A_agent_food.append([f[i][1] for i in range(n)])
    #print(A_agent_food)#[[33, 41, 16, 31], [22, 37, 31, 39], [22, 37, 31, 39], [20, 0, 8, 1], [26, 46, 10, 47]]

    A_agent_food_ = []
    for i, af in enumerate(A_agent_food):
        a = [[], []]
        for f in af:
            if ((foodInfos[f][0] - agentInfos[A_agent[i][0]][0]) ** 2 + (foodInfos[f][1] - agentInfos[A_agent[i][0]][1]) ** 2) - ((foodInfos[f][0] - agentInfos[A_agent[i][1]][0]) ** 2 + (foodInfos[f][1] - agentInfos[A_agent[i][1]][1]) ** 2) < 0:
                a[0].append(f)
            else:
                a[1].append(f)
        A_agent_food_.append(a)
    # print(A_agent_food_) #[[[33, 41], [16, 31]], [[22, 37, 31], [39]], [[39], [22, 37, 31]], [[20], [0, 8, 1]], [[26, 46, 10, 47], []]]


    A_agent_badfood = []
    for a in A_agent:
        f = []
        for r in range(len(badFoodsInfos)):
            f.append([
                (badFoodsInfos[r][0] - agentInfos[a[0]][0]) ** 2 + (badFoodsInfos[r][1] - agentInfos[a[0]][1]) ** 2 +
                (badFoodsInfos[r][0] - agentInfos[a[1]][0]) ** 2 + (badFoodsInfos[r][1] - agentInfos[a[1]][1]) ** 2
                , r
            ])
        f.sort(key=lambda x: x[0])
        A_agent_badfood.append([f[i][1] for i in range(n)])
    # print(A_agent_badfood)#[[33, 41, 16, 31], [22, 37, 31, 39], [22, 37, 31, 39], [20, 0, 8, 1], [26, 46, 10, 47]]

    A_agent_badfood_ = []
    for i, af in enumerate(A_agent_badfood):
        a = [[], []]
        for f in af:
            if ((badFoodsInfos[f][0] - agentInfos[A_agent[i][0]][0]) ** 2 + (badFoodsInfos[f][1] - agentInfos[A_agent[i][0]][1]) ** 2) - ((badFoodsInfos[f][0] - agentInfos[A_agent[i][1]][0]) ** 2 + (badFoodsInfos[f][1] - agentInfos[A_agent[i][1]][1]) ** 2) < 0:
                a[0].append(f)
            else:
                a[1].append(f)
        A_agent_badfood_.append(a)
    # print(A_agent_badfood_) #[[[33, 41], [16, 31]], [[22, 37, 31], [39]], [[39], [22, 37, 31]], [[20], [0, 8, 1]], [[26, 46, 10, 47], []]]

    X = []
    A = []
    for i in range(len(A_agent)):
        index = []
        index_relation = []
        agent = A_agent[i]
        a = A_agent_food_[i]
        b = A_agent_badfood_[i]
        for j in agent:
            index.append(j)
        for j in a:
            if len(j) > 0:
                for s in j:
                    index.append(s+5)
        for j in b:
            if len(j) > 0:
                for s in j:
                    index.append(s+55)
        index_relation.append(agent)
        for j in range(len(agent)):
            if len(a[j]) > 0:
                for s in a[j]:
                    index_relation.append([agent[j], s+5])
            if len(b[j]) > 0:
                for s in b[j]:
                    index_relation.append([agent[j], s+55])

        A_ = np.zeros(shape=(len(index), len(index)))
        X_ = []
        for relation in index_relation:
            A_[index.index(relation[0])][index.index(relation[1])] = 1
            A_[index.index(relation[1])][index.index(relation[0])] = 1
        for ind in index:
            X_.append(obs_[ind])
        A.append(A_)
        X.append(X_)
    return np.array(A), np.array(X)

if __name__ == '__main__':
    env = UnityEnvironment()
    obs = env.reset(train_mode=True)
    brain_name = env.brain_names[0]
    obs = obs[brain_name].vector_observations
    obs = obs[0]
    A, X = preprocess_observation(obs)
    print(A)
    print(X)
    env.close()

