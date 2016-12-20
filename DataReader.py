__author__ = 'quynhdo'
import numpy as np
class SimpleSemanticFrame:
    def __init__(self, sentence, pred_id, labels):
        '''
        define a simple semantic frame with predicate and argument labels
        :param sentence: the sentence, for example, 'he ran fast'
        :param pred_id: the position of the predicate, in the above example, it is 1
        :param labels: the labels of the words in the sentence, should be a Python List, in the above example: ["A1", None, "AM-MNR"]
        :return:
        '''
        self.words = sentence.split()  # words in the sentence
        self.pred = self.words[pred_id] # predicate word
        self.labels = labels # labels of the words in sentence

    def extract_instance(self):
        '''
        extract instances for argument classification training, each instance is a triple of (Word, Predicate, Label)
        :return:
        '''
        instances = []
        for w_id in range(len(self.words)):
            if self.labels[w_id] != None:
                instances.append((self.words[w_id], self.pred, self.labels[w_id]))
        return instances


class DataSet:
    '''
    Dataset for machine learning algorithm, it should store argument classification instances, and extract features to process a machine learning algorithm
    '''
    def __init__(self):
        self.data = [] # instances
        self.labels = [] # label vocabulary
        self.fea_arg_form = [] # vocabulary of argument word form feature
        self.fea_pred_form = [] # vocabulary of predicate word form feature

    def add_instances(self, instances):
        self.data.append(instances)

    def extract_labels(self):
        '''
        extract label vocabulary
        :return:
        '''
        for instances in self.data:
            for ins in instances:
                if not ins[2] in self.labels:
                    self.labels.append(ins[2])



    def extract_features(self):
        '''
        extract feature vocabularies
        :return:
        '''

        for instances in self.data:
            for ins in instances:
                if ins[0] not in self.fea_arg_form:
                    self.fea_arg_form.append(ins[0])
                if ins[1] not in self.fea_pred_form:
                    self.fea_pred_form.append(ins[1])






    def get_representation(self, instance, prev_instance = "null", use_un_observed=True):
        '''
        get representation
        :param instance:
        :param prev_instance: previous instance, if the current instance is the first one, then the prev_instance = null
        :param use_un_observed: use unobserved feature or not
        :return:
        '''
        rep =[]
        fea=[]
        # argument word form features
        for arg in self.fea_arg_form:
            for lbl in self.labels:
                fea.append((arg,lbl))
                if instance[0]==arg and instance[2]== lbl:
                    rep.append(1)
                else:
                    rep.append(0)

        # predicate word form features
        for pred in self.fea_pred_form:
            for lbl in self.labels:
                fea.append((pred,lbl))

                if instance[1]==pred and instance[2]== lbl:
                    rep.append(1)
                else:
                    rep.append(0)


        # edge features: prev_label - current_label
        for lbl in self.labels + ["null"]:
            for lbl2 in self.labels:
                if lbl != lbl2:
                    fea.append((lbl,lbl2))
                    if prev_instance == lbl and instance[2]== lbl2:
                        rep.append(1)
                    else:
                        rep.append(0)

        self.feas = fea

        if not use_un_observed:
            rs = []
            for i in range(len(rep)):
                if not i in self.unobserved_feas:
                    rs.append(rep[i])
            rep=rs

        return rep


    def get_all_preresentations(self, instances, use_unobserved=True):
        '''
        get representations for a list of instances
        :param instances:
        :param use_unobserved:
        :return:
        '''
        reps = []
        for i in range(len(instances)):
            if i==0:
                reps.append(self.get_representation(instances[i], "null", use_unobserved))
            else:
                reps.append(self.get_representation(instances[i],instances[i-1][2] , use_unobserved))

        return np.asarray(reps, dtype="float64")

    def not_in_data(self):
        '''
        identify unobserved features
        :return:
        '''
        self.unobserved_feas = []  # stores the indexes of unobserved features
        for f in self.feas:
            observed = False
            for instances in self.data:
                for ins in instances:
                    if f == (ins[0], ins[2]) or f == (ins[1],ins[2]):
                        observed=True
                    if instances.index(ins)==0:
                        if f == ("null", ins[2]):
                            observed=True
                    else:
                        if f == (instances[instances.index(ins)-1][2], ins[2]):
                            observed=True
            if not observed:
                self.unobserved_feas.append(self.feas.index(f))
