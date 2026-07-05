from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
import random
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from pyitlib import discrete_random_variable as drv

def find_common_elements(lists):
    """
    Taking consecutive intersections of sets provided by the input list, return as list
    """
    common_elements = set(lists[0])
    for sublist in lists[1:]:
        common_elements = common_elements.intersection(sublist)
    
    return list(common_elements)
    
    
def find_union(lists):
    """
    Taking consecutive union of sets provided by the input list, return as list
    """
    merged_list = [item for sublist in lists for item in sublist]

    union_elements = set(merged_list)

    return list(union_elements)
    
    
def find_difference(list1, list2):
    """
    Taking difference set of list1 over list2, return as list 
    """
    set1 = set(list1)
    set2 = set(list2)

    difference = set1 - set2
    return list(difference)
    
    
def fa(k, a_set, v_set, sim, row, col):
    if len(a_set) == 0:
        init_a_set = []
        marginal_v = 0
        for i in v_set:
            max_ki = 0
            if k == col[i]:
                max_ki = sim[i]
            init_a_set.append(max_ki)
            marginal_v += max_ki
        return marginal_v, init_a_set

    new_a_set = []
    marginal_v = 0
    for i in v_set:
        sim_ik = 0
        if k == col[i]:
            sim_ik = sim[i]

        if sim_ik > a_set[i]:
            max_ki = sim_ik
            new_a_set.append(max_ki)
            marginal_v += max_ki - a_set[i]
        else:
            new_a_set.append(a_set[i])
    return marginal_v, new_a_set
    
def lazier_and_goals_sample_kg(goals, batch_size_in_transitions=100, sub_size = 5):

    num_neighbor = 1
    kgraph = NearestNeighbors(
            n_neighbors=num_neighbor, algorithm='kd_tree',
            metric='euclidean').fit(goals).kneighbors_graph(
                mode='distance').tocoo(copy=False)
    row = kgraph.row
    col = kgraph.col
    sim = np.exp(
            -np.divide(np.power(kgraph.data, 2),
                       np.mean(kgraph.data)**2))
    delta = np.mean(kgraph.data)

    sel_idx_set = []
    idx_set = [i for i in range(len(goals))]
    balance = -1
    if int(balance) == -1:
        balance = math.pow(
                1 + 0.0001,
                1) * 1
    v_set = [i for i in range(len(goals))]
    max_set = []
    for i in range(0, batch_size_in_transitions):
        sub_set = random.sample(idx_set, sub_size)
        sel_idx = -1
        max_marginal = float("-inf")  #-1 may have an issue
        for j in range(sub_size):
            k_idx = sub_set[j]
            marginal_v, new_a_set = fa(k_idx, max_set, v_set, sim, row,
                                           col)
            # euc = np.linalg.norm(goals[sub_set[j]] - ac_goals[sub_set[j]])
            # marginal_v = marginal_v - balance * euc
            marginal_v = -marginal_v
            if marginal_v > max_marginal:
                sel_idx = k_idx
                max_marginal = marginal_v
                max_set = new_a_set

        idx_set.remove(sel_idx)
        sel_idx_set.append(sel_idx)
    return np.array(sel_idx_set)
    
    
def multiply_hash_array(array, multiplier=100):
    """
        Discretize the array by multiplying the multiplier and integrating it.
    """
    hashed_array = np.zeros_like(array, dtype=int)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            hashed_array[i, j] = int(array[i, j] * multiplier)
    return hashed_array


def min_count_value_and_index(array, states, arrival_ratio=0.5):
    new_counter = Counter(map(tuple, array))

    for key, value in new_counter.items():
        if key in self.counter:
            self.counter[key] += value
        else:
            self.counter[key] = value
                
            index = [i for i in range(array.shape[0]) if np.array_equal(array[i], key)]
            self.c_state[key] = states[index[0]]
                
    min_count = min(self.counter.values())
    min_subarrays = [subarray for subarray, count in self.counter.items() if count == min_count]
        
    return_state = np.zeros((len(min_subarrays), self.state_dim))
    for i in range(len(min_subarrays)):
        return_state[i] =  self.c_state[min_subarrays[i]]
            
        
    for k in min_subarrays:
        if k in self.counter:
            self.counter[k] += arrival_ratio
        else:
            raise KeyError(f"The key '{key}' does not exist in self.counter.")


    return return_state, min_subarrays, min_count




def convert_to_one_hot(array, num_classes):
    one_hot_array = np.zeros((array.size, num_classes), dtype=np.int)
    for i in range(array.size):
        one_hot_array[i, array[i]] = 1
    return one_hot_array
    


def get_d4actions_yanchi(tensor_s, tensor_a, threshold, agent_id=0, num_classes=0, scaler=100):
    numpy_s = tensor_s.numpy() # (batch_size, episode_len, state_size)
    numpy_a = tensor_a.numpy() # (batch_size, episode_len, agent_num, 1)
    # print(numpy_s.shape)
    # print(numpy_a.shape)
    
    caus_a = numpy_a[:, agent_id, :]
    caus_a_onehot = convert_to_one_hot(caus_a, num_classes)
    
    state_shape = numpy_s.shape[-1]
    
        
    s_diff_arr = numpy_s
    # for a_one_hot in a_onehot_list:
    #     print(a_one_hot.shape)
    a_onehot_arr = caus_a_onehot


    indices4allactions = []
    values4allactions = []
    
    ae_list = []
    for action_idx in range(a_onehot_arr.shape[-1]):
        mi_list = []
        for state_idx in range(state_shape):
            x = s_diff_arr[:, state_idx].reshape(-1)
            # x = (np.round(x, 2) * 100).astype(int)
            x = (np.round(x, 2) * 20).astype(int)
            
            # raise NotImplementedError
            
            y = a_onehot_arr[:, action_idx].astype(int)
            
            mi = normalized_mutual_info_score(x, y)
            mi_list.append(mi)
        ae_list.append(mi_list)
        mi_list = np.array(mi_list)
        # print("action id: {}".format(action_idx))
        indices = [index for index, value in enumerate(mi_list) if value > threshold]
        values = [value for index, value in enumerate(mi_list) if value > threshold]
        
        indices4allactions.append(indices)
        values4allactions.append(values)
        # print("index:", indices)
        # print(mi_list)
    
    return indices4allactions, values4allactions    
    
