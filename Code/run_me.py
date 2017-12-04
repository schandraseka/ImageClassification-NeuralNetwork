# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter
from itertools import product

# Read in train and test data
def read_image_data():
    print('Reading image data ...')
    temp = np.load('../../Data/data_train.npz')
    train_x = temp['data_train']
    temp = np.load('../../Data/labels_train.npz')
    train_y = temp['labels_train']
    temp = np.load('../../Data/data_test.npz')
    test_x = temp['data_test']
    return (train_x, train_y, test_x)

############################################################################

def parse_param_grid(param_grid):
    for p in param_grid:
            items = sorted(p.items())
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params


def gridsearch(X, y, model, paramgridIterator, cv=5):
    result = []
    maxacc = 0
    bestparam = {}
    accuracies = []
    for param in paramgridIterator:
        model = model.set_params(**param)
        accuracies = cross_val_score(model,X,y,scoring='accuracy',cv=5)
        acc = np.mean(accuracies)
        if abs(acc-maxacc)>0:
            maxacc = acc
            bestparam = param
        print(str(param)+"Scores="+str(acc))
        print('Error', 1-acc)
    return bestparam,maxacc


train_x, train_y, test_x = read_image_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)


param_grid = {'max_depth' : [3, 6, 9, 12, 14
]}
paramgridIterator = parse_param_grid([param_grid])
model = DecisionTreeClassifier()
bestparam = gridsearch(train_x, train_y, model, paramgridIterator)[0]
print('Best parameters',bestparam)
bestmodel = DecisionTreeClassifier(max_depth = bestparam['max_depth'])
test_y = bestmodel.predict(test_x)
file_name = '../Predictions/destree.csv'
# Writing output in Kaggle format    
print('Writing output to ', file_name)
kaggle.kaggleize(test_y, file_name)

param_grid = {'n_neighbors' : [3, 5, 7, 9, 11]}
paramgridIterator = parse_param_grid([param_grid])
model = KNeighborsClassifier()
bestparam = gridsearch(train_x, train_y, model, paramgridIterator)[0]
print('Best parameters',bestparam)
bestmodel = KNeighborsClassifier(n_neighbors = bestparam['n_neighbors'])
test_y = bestmodel.predict(test_x)
#predicted_y = np.random.randint(0, 4, test_x.shape[0])
#print('DUMMY Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))
# Output file location
file_name = '../Predictions/knn.csv'
# Writing output in Kaggle format    
print('Writing output to ', file_name)
kaggle.kaggleize(test_y, file_name)






param_grid = {"loss" : ["hinge", "log"], "alpha" : [1e-6 , 1e-4 , 1e-2 , 1, 10]}
paramgridIterator = parse_param_grid([param_grid])
model = linear_model.SGDClassifier()
bestparam = gridsearch(train_x, train_y, model, paramgridIterator)[0]
bestmodel = linear_model.SGDClassifier(alpha = bestparam['alpha'], loss=bestparam['loss'], penalty ='l2', shuffle=True, random_state=None)
print('Best parameters',bestparam)
bestmodel.fit(train_x,train_y)
test_y = bestmodel.predict(test_x)
#predicted_y = np.random.randint(0, 4, test_x.shape[0])
#print('Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))
# Output file location
file_name = '../Predictions/sgd.csv'
# Writing output in Kaggle format    
print('Writing output to ', file_name)
kaggle.kaggleize(test_y, file_name)


