# importing module
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

"""
#Tuning and evaluation of models
def evalModel(model, X_test, y_test, y_pred_class):
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    # print("Accuracy: ", acc_score)
    # print("NULL Accuracy: ", y_test.value_counts())
    # print("Percentage of ones: ", y_test.mean())
    # print("Percentage of zeros: ", 1 - y_test.mean())

    #creating a confunsion matrix
    conmat = metrics.confusion_matrix(y_test, y_pred_class)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    sns.heatmap(conmat, annot=True)
    plt.title("Confusion " + str(model))
    plt.xlabel("predicted")
    plt.ylabel("Actual")
    # plt.show()
    plt.savefig("output_graph/Confusion_" + str(model).partition("(")[0] + ".png")
    return acc_score
"""


#creating a confunsion matrix
def evalModel(model, X_test, y_test, y_pred_class):
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    #creating a confunsion matrix
    conmat = metrics.confusion_matrix(y_test, y_pred_class)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    plt.figure()
    heatmap = sns.heatmap(conmat, annot=True, cbar=False)
    heatmap = heatmap.get_figure()
    plt.title("Confusion " + str(model))
    plt.xlabel("predicted")
    plt.ylabel("Actual")
    # plt.show()
    temp = "D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/Confusion_" + str(model).partition("(")[0] + ".png"
    heatmap.savefig(temp)
    print("output_graph/Confusion_" + str(model).partition("(")[0] + ".png")
    return acc_score