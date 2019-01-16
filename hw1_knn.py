from __future__ import division, print_function

from typing import List, Callable

import numpy as np
import scipy
from collections import Counter


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: Complete the training function
    def train(self, features: List[List[float]], labels: List[int]):
        self.train_features = features
        self.train_labels = labels
        
    
    #TODO: Complete the prediction function
    def predict(self, features: List[List[float]]) -> List[int]:
        prediction_labels = []
        for i in range(len(features)):
            k_neighbore_lables = self.get_k_neighbors(features[i])
            prediction_labels.append(self.get_near_by_majority_label(k_neighbore_lables))
        return prediction_labels
        
    #TODO: Complete the get k nearest neighbor function
    def get_k_neighbors(self, point):
        nearestNeighbors = []
        kNeighborsLabels = []
        for idx in range(len(self.train_features)):
            dist_value = self.distance_function(point, self.train_features[idx])
            nearestNeighbors.append([dist_value, self.train_labels[idx]])

        nearestNeighbors.sort(key=lambda idx: idx[0])
        count=0
        while(count<self.k):
            kNeighborsLabels.append(nearestNeighbors[count][1])
            count = count+1   
        return kNeighborsLabels
    
    def get_near_by_majority_label(self, k_neighbore_lables) -> int:
        negative_label = 0
        positive_label = 0
        for x in range(len(k_neighbore_lables)):
            if(k_neighbore_lables[x] == 0):
                negative_label = negative_label+1
            else:
                positive_label = positive_label+1

        if(negative_label>positive_label):
            return 0
        else:
            return 1
        
        
    #TODO: Complete the model selection function where you need to find the best k     
    def model_selection_without_normalization(distance_funcs, train_features, train_labels, f1_score, valid_features, valid_labels, test_features, test_labels):
        
            
            #Dont change any print statement
            for name, func in distance_funcs.items():
                best_f1_score, best_k = -1, 0
                for k in [1, 3, 10, 20, 50]:
                    model = KNN(k=k, distance_function=func)
                    model.train(train_features, train_labels)
                    train_f1_score = f1_score(train_labels,  model.predict(train_features))
                    valid_f1_score = f1_score(valid_labels, model.predict(valid_features))
                    
                    
                    print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
            
                    if valid_f1_score > best_f1_score:
                        best_f1_score, best_k,model = valid_f1_score, k,model

                model = KNN(k=best_k, distance_function=func)
                model.train(train_features,train_labels)
                test_f1_score = f1_score(test_labels, model.predict(test_features))
                print()

                print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
                          'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
                print()
            return best_k, model
    
    #TODO: Complete the model selection function where you need to find the best k with transformation
    def model_selection_with_transformation(distance_funcs,scaling_classes, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        
                #Dont change any print statement
                print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
                          'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) + 
                          'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
    
                print()
                print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
                      'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
                print()
        
        
    #TODO: Do the classification 
    def test_classify(model):
        print()
        

if __name__ == '__main__':
    
    print(numpy.__version__)
    print(scipy.__version__)
