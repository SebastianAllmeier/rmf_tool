import src.rmf_tool as rmf
import numpy as np
import pickle

def dChoiceModel(K, rho, d):
    ddpp = rmf.DDPP()

    # The vector 'e(i)' is a vector where the $i$th coordinate is equal to $1$ (the other being equal to $0$)
    def e(i):
        l = np.zeros(K)
        l[i] = 1
        return(l)

    # We then add the transitions : 
    for i in range(K):
        if i>=1:
            ddpp.add_transition(e(i),eval('lambda x: {}*(x[{}]**{} - x[{}]**{} )'.format(rho,i-1,d,i,d) ))
        if i<K-1:
            ddpp.add_transition(-e(i),eval('lambda x: (x[{}] - x[{}])'.format(i,i+1) ))
    ddpp.add_transition(e(0), lambda x : eval('{}*(1-x[0]**{})'.format(rho,d)))
    ddpp.add_transition(-e(K-1), lambda x : x[K-1])
    ddpp.set_initial_state(e(0))
    return ddpp

def generate_data():
    data = dict([])
    for rho in [0.6,0.7,0.8,0.9]:
        for d in [2,3]:
            for K in [5,9,15,20]:
                for order in ([1, 2] if K<=5 else [1]):
                    ddpp = dChoiceModel(K, rho, d)
                    data[(K,rho,d,order)] = ddpp.meanFieldExpansionSteadyState(order=order)
    with open('output_tests/d_choice.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def approximately_equal(new_data, old_data):
    absolute_difference = 0
    for i in [0,1] if len(new_data)==3 else [0,1,2]:
        new = np.array(new_data[i])
        old = np.array(old_data[i])
        absolute_difference += np.sum(np.abs(new-old))
    return absolute_difference
    
def test_two_choice():
    with open('output_tests/d_choice.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        for key in data:
            (K,rho,d,order) = key
            print(key)
            ddpp = dChoiceModel(K, rho, d)
            new_data =  ddpp.meanFieldExpansionSteadyState(order=order)
            test_data = data[key]
            assert(approximately_equal(new_data, test_data) <= 1e-8)

#generate_data()
#test_two_choice()
