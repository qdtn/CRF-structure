import itertools
import numpy as np
from DataReader import DataSet
from DataReader import SimpleSemanticFrame

__author__ = 'quynhdo'

class CRF:
    '''
    A simple CRF implementation
    '''

    def __init__(self, num_feas, labels, lr=0.1, un_observed=0):
        '''

        :param num_feas: number of features
        :param labels: label vocabulary
        :param lr: learning rate
        :param un_observed: number of unobserved features
        :return:
        '''
        self.weights=np.zeros(num_feas-un_observed, dtype="float64")
        self.labels = labels
        self.lr = lr
        self.num_feas=num_feas

    def train(self, instances,  ds, use_unobserved=True):
        num_ins =len(instances)
        L_prime = []


        for lprime in list( itertools.permutations(self.labels, num_ins)):
            instances_prime = []
            for i in range(num_ins):
                ins = (instances[i][0], instances[i][1], lprime[i])
                instances_prime.append(ins)
            L_prime.append(instances_prime)

        #print (L_prime)

        score = self.get_score(ds.get_all_preresentations(instances,use_unobserved))

        scores_prime =  [self.get_score(ds.get_all_preresentations(lprime, use_unobserved)) for lprime in L_prime]
        total_scores= np.sum(np.asarray(scores_prime))


        grad = np.sum(ds.get_all_preresentations(instances, use_unobserved), axis=0)

        for i in range(len( L_prime)):
            rp = ds.get_all_preresentations(L_prime[i],  use_unobserved)
            #print (scores_prime)
            #print (total_scores)
            #print (rp)
            #print (np.sum(rp, axis=0))
            #print (np.sum(rp, axis=0) * scores_prime[i]  / total_scores)
            grad -= np.sum(rp, axis=0) * scores_prime[i]  / total_scores

            #print (grad)

        self.weights = self.weights + self.lr * grad
        print ("Weights:")
        print (self.weights)




    def get_score(self, reps):
        return np.exp( np.sum(np.sum(reps, axis=0) * self.weights ))



    def generate_possible_labels(self, n):
        return itertools.combinations(self.labels, n)


if __name__=="__main__":
    '''
    Use all unobserved features
    '''
    ds = DataSet()


    fr = SimpleSemanticFrame("he ran fast",1, ["A1", None, "AM-MNR"])
    ds.add_instances(fr.extract_instance())

    ds.extract_labels()
    ds.extract_features()



    print (ds.get_representation(ds.data[0][0]))
    ds.not_in_data()
    #print (len(ds.get_representation(ds.data[0][0])))
    #print (ds.feas)
    print (ds.unobserved_feas)
    crf = CRF (len(ds.feas), ds.labels)#, un_observed=len(ds.unobserved_feas))

    #print (ds.get_all_preresentations(ds.data[0]))

    #print (crf.weights)

    crf.train(ds.data[0], ds)


    crf.train(ds.data[0], ds)




    crf.train(ds.data[0], ds)


    crf.train(ds.data[0], ds)





    '''
    Don't use unobserved features
    '''
    ds = DataSet()


    fr = SimpleSemanticFrame("he ran fast",1, ["A1", None, "AM-MNR"])
    ds.add_instances(fr.extract_instance())

    ds.extract_labels()
    ds.extract_features()



    print (ds.get_representation(ds.data[0][0]))
    ds.not_in_data()
    #print (len(ds.get_representation(ds.data[0][0])))
    #print (ds.feas)
    print (ds.unobserved_feas)
    crf = CRF (len(ds.feas), ds.labels, un_observed=len(ds.unobserved_feas))

    #print (ds.get_all_preresentations(ds.data[0]))

    #print (crf.weights)

    crf.train(ds.data[0], ds, use_unobserved=False)


    crf.train(ds.data[0], ds, use_unobserved=False)


    crf.train(ds.data[0], ds, use_unobserved=False)



    crf.train(ds.data[0], ds, use_unobserved=False)

