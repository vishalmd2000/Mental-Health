#time logging library
import time

# importing from src
from src.ModelEvaluation import evalModel
from src.output import get_csv_output

# sklearn module for tuning
# from sklearn.model_selection import GridSearchCV
import pickle


# sklearn modules for model creation
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier


with open("log_pkl", "br") as log,\
     open("dt_pkl", "br") as dt,\
     open("rf_pkl", "br") as rf,\
     open("knn_pkl", "br") as knn,\
     open("stac_pkl", "br") as stack,\
     open("bag_pkl", "br") as bag,\
     open("boost_pkl", "br") as boost:
    log = pickle.load(log)
    dt = pickle.load(dt)
    rf = pickle.load(rf)
    knn = pickle.load(knn)
    stack = pickle.load(stack)
    bag = pickle.load(bag)
    boost = pickle.load(boost)
        


# Run all model in one shot
def GridSearch(X_train, X_test, y_train, y_test, accuracyDict, timelog):
    start = time.time()
    log_reg_mod(X_train, X_test, y_train, y_test, accuracyDict)
    tuneKNN(X_train, X_test, y_train, y_test, accuracyDict)
    tuneDT(X_train, X_test, y_train, y_test, accuracyDict)
    tuneRF(X_train, X_test, y_train, y_test, accuracyDict)
    tuneBoosting(X_train, X_test, y_train, y_test, accuracyDict)
    tuneBagging(X_train, X_test, y_train, y_test, accuracyDict)
    tuneStacking(X_train, X_test, y_train, y_test, accuracyDict)
    end = time.time()
    timelog['GridSearch models'] = end - start

# tuning the logistic regression model with Gridsearchcv
def log_reg_mod(X_train, X_test, y_train, y_test, accuracyDict):
    # global lr
    print("\nTuning the Logistic Regression Model with GridSearchCV\n")
    # param_grid = {'C':[0.1,1,10,100,1000],
    #               'solver':['newton-cg','lbfgs','sag'],
    #               'multi_class':['ovr','multinomial'],
    #               'max_iter':[100,200,300,400,500]}
    # grid_search = GridSearchCV(LogisticRegression(), param_grid, n_jobs=-1,  cv=5)
    # grid_search.fit(X_train,y_train)
    # # print("Best param_grids: ", grid_search.best_params_)
    # # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # # print("Best estimator: ", grid_search.best_estimator_)
    # lr = grid_search.best_estimator_
    y_pred_class = log.predict(X_test)
    accuracy = evalModel(log, X_test, y_test, y_pred_class)
    accuracyDict['Log_Reg_mod_GSCV'] = accuracy * 100
    get_csv_output(log, X_test, y_pred_class)

# tuning the KNN model with GridSearchCV
def tuneKNN(X_train, X_test, y_train, y_test, accuracyDict):
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
    y_pred_class = knn.predict(X_test)
    accuracy = evalModel(knn,X_test, y_test, y_pred_class)
    accuracyDict['KNN_GSCV'] = accuracy * 100
    get_csv_output(knn, X_test, y_pred_class)

# tuning the Decision Tree model with GridSearchCV
def tuneDT(X_train, X_test, y_train, y_test, accuracyDict):
    # print("\nTuning Decision Tree model with GridSearchCV\n")
    # param_grid = {'criterion':['gini','entropy'],'max_depth':[3,5,7,9,11,13,15],
    #               'min_samples_split':[2,3,4,5,6,7,8],'random_state':[0]}
    # grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, n_jobs=-1,  cv=5)
    # grid_search.fit(X_train,y_train)
    # # print("Best param_grids: ", grid_search.best_params_)
    # # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # # print("Best estimator: ", grid_search.best_estimator_)
    # dt = grid_search.best_estimator_
    y_pred_class = dt.predict(X_test)
    accuracy = evalModel(dt,X_test, y_test, y_pred_class)
    accuracyDict['DecisionTree_GSCV'] = accuracy * 100
    get_csv_output(dt, X_test, y_pred_class)

# tuning the Random Forest model with GridSearchCV
def tuneRF(X_train, X_test, y_train, y_test, accuracyDict):
    # global rf
    # print("\nTuning Random Forest model with GridSearchCV\n")
    # param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],'max_depth':[3,5,7,9,11,13,15],
    #         'min_samples_split':[2,3,4,5,6,7,8],'criterion':['gini','entropy'],'random_state':[0]}
    # grid_search = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1,  cv=5)
    # grid_search.fit(X_train,y_train)
    # # print("Best param_grids: ", grid_search.best_params_)
    # # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # # print("Best estimator: ", grid_search.best_estimator_)
    # rf = grid_search.best_estimator_
    y_pred_class = rf.predict(X_test)
    accuracy = evalModel(rf,X_test, y_test, y_pred_class)
    accuracyDict['RandomForest_GSCV'] = accuracy * 100
    get_csv_output(rf, X_test, y_pred_class)

# tuning boosting model with GridSearchCV
def tuneBoosting(X_train, X_test, y_train, y_test, accuracyDict):
    # print("\nTuning Boosting model with GridSearchCV\n")
    # param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
    #               'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],'random_state':[0]}
    # grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, n_jobs=-1,  cv=5)
    # grid_search.fit(X_train,y_train)
    # # print("Best param_grids: ", grid_search.best_params_)
    # # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # # print("Best estimator: ", grid_search.best_estimator_)
    # ada = grid_search.best_estimator_
    y_pred_class = boost.predict(X_test)
    accuracy = evalModel(boost,X_test, y_test, y_pred_class)
    accuracyDict['BoostBoost_GSCV'] = accuracy * 100
    get_csv_output(boost, X_test, y_pred_class)

# tuning bagging model with GridSearchCV
def tuneBagging(X_train, X_test, y_train, y_test, accuracyDict):
    # print("\nTuning Bagging model with GridSearchCV\n")
    # param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
    #               'max_samples':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    #               'bootstrap':[True,False],'bootstrap_features':[True,False],
    #               'random_state':[0]}
    # grid_search = GridSearchCV(BaggingClassifier(), param_grid, n_jobs=-1,  cv=5)
    # grid_search.fit(X_train,y_train)
    # # print("Best param_grids: ", grid_search.best_params_)
    # # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # # print("Best estimator: ", grid_search.best_estimator_)
    # bag = grid_search.best_estimator_
    y_pred_class = bag.predict(X_test)
    accuracy = evalModel(bag,X_test, y_test, y_pred_class)
    accuracyDict['Bagging_GSCV'] = accuracy * 100
    get_csv_output(bag, X_test, y_pred_class)
    
# tuning stacking model with GridSearchCV
def tuneStacking(X_train, X_test, y_train, y_test, accuracyDict):
    # classifiers=[('rf',rf),('lr', lr),('knn', knn)]
    # print("\nTuning Stacking model with GridSearchCV\n")
    # param_grid = {'stack_method': ['predict_proba', 'decision_function', 'predict']}
    # grid_search = GridSearchCV(StackingClassifier(estimators = classifiers), param_grid, n_jobs=-1,  cv=5)
    # grid_search.fit(X_train,y_train)
    # # print("Best param_grids: ", grid_search.best_params_)
    # # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # # print("Best estimator: ", grid_search.best_estimator_)
    # stack = grid_search.best_estimator_
    y_pred_class = stack.predict(X_test)
    accuracy = evalModel(stack,X_test,y_test, y_pred_class)
    accuracyDict['Stacking_GSCV'] = accuracy * 100
    get_csv_output(stack, X_test, y_pred_class)