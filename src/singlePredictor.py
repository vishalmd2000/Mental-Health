import pandas as pd
import pickle
# from output import get_csv_output

         
def tuneKNN(X_test):
    # global knn
    # print("\nTuning KNN model with GridSearchCV\n")
    # param_grid = {'n_neighbors':[3,5,7,9,11,13,15],
    #               'weights':['uniform','distance'],
    #               'algorithm':['auto','ball_tree','kd_tree','brute'],
    #               'leaf_size':[10,20,30,40,50,60,70,80]}
    # grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1,  cv=5)
    # grid_search.fit(X_train,y_train)
    # # print("Best param_grids: ", grid_search.best_params_)
    # # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # # print("Best estimator: ", grid_search.best_estimator_)
    # knn = grid_search.best_estimator_
    with open("knn_pkl", "br") as knn:
        knn = pickle.load(knn)

    y_pred_class = knn.predict(X_test)
    # accuracy = evalModel(knn,X_test, y_test, y_pred_class)
    # accuracyDict['KNN_GSCV'] = accuracy * 100
    # get_csv_output(knn, X_test, y_pred_class)
    return y_pred_class

# dic = {
#     "family_size": 1,
#     "annual_income": 3,
#     "eating_habits": 2,
#     "addiction_friend": 2,
#     "addiction": 1,
#     "medical_history": 0,
#     "depressed": 0,
#     "anxiety": 0,
#     "happy_currently": 1,
#     # "suicidal_thoughts": 0
# }
# dic = pd.DataFrame(dic, index=[0])
# x = tuneKNN(dic)
# print(x[0])
# print(type(x))
