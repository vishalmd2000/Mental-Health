import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# function to get CSV output of our predictions 
# def get_csv_output(model, X_test, y_pred_class):
#     name = str(model).partition('(')
#     y_pred = pd.Series(y_pred_class, name='predictions')
#     measure = pd.Series(X_test['depressed'], name='Measure')
#     measure = pd.DataFrame(measure)
#     output_data = measure.join(y_pred)
#     csv = pd.DataFrame(output_data)
#     path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/output_result/"
#     file_name = path + name[0] +'.csv'
#     csv.to_csv(file_name, header=True)

def get_csv_output(model, y_test, y_pred_class):
    name = str(model).partition('(')
    csv=0
    y_pred = pd.Series(y_pred_class, name='predictions')
    print(y_pred)
    print()
    print(type(y_test))
    measure = y_test.reset_index(drop=True)
    print(measure)
    # measure = pd.DataFrame(measure)
    # output_data = measure.join(y_pred)
    output_data = pd.concat([measure, y_pred], axis=1)
    csv = pd.DataFrame(output_data)
    # path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/output_result/"
    path = "output_result/"
    file_name = path + name[0] +'.csv'
    csv.to_csv(file_name, header=True)


def get_pdf_output():
    df = pd.read_csv("output_result/KNeighborsClassifier.csv")
    df['predictions'].loc[(df.predictions == 1)] = "maybe"
    df['predictions'].loc[(df.predictions == 0)] = "maybe not"
    plt.clf()
    table = plt.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig("output_graph/table.png")
    file_name = "Report.pdf"

    image1 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/FeatureImportance.png").convert('RGB')
    image2 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/CorrelationMatrix.png").convert('RGB')
    image3 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/AccuracyBarGraph.png").convert('RGB')
    image4 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/Confusion_LogisticRegression.png").convert('RGB')
    image5 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/Confusion_DecisionTreeClassifier.png").convert('RGB')
    image6 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/Confusion_RandomForestClassifier.png").convert('RGB')
    image7 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/Confusion_KNeighborsClassifier.png").convert('RGB')
    image8 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/Confusion_StackingClassifier.png").convert('RGB')
    image9 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/Confusion_BaggingClassifier.png").convert('RGB')
    image10 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/Confusion_AdaBoostClassifier.png").convert('RGB')
    image11 = Image.open("D:/Engineering/Project/5/Mental_health/Mental_health/output_graph/table.png").convert('RGB')
    image_list = [image2, image3, image4, image5, image6, image7, image8, image9, image10, image11]

    image1.save(file_name, save_all=True, append_images=image_list)
    print("Saved pdf")


def visual_final_plot(log, kn, dis, rand, boosting, bagging):
    X_axis = ['log_reg', 'knn', 'dictree', 'rand_for', 'boost', 'bag']
    bar_width = np.arange(6)
    actual_vals = [log[0], kn[0], dis[0], rand[0], boosting[0], bagging[0]]
    predicted_vals = [log[1], kn[1], dis[1], rand[1], boosting[1], bagging[1]]
    plt.bar(bar_width, actual_vals, width =0.45, align='edge', label="Actual Values")
    plt.bar(bar_width + 0.45, predicted_vals,width=0.45, align='edge', label="Predicted Values")
    # plt.bar(kn,height=21, label="KNN")
    # plt.bar(kn,height=21, label="KNN")
    # plt.bar(dis,height=21, label="Dicision")
    # plt.bar(dis,height=21, label="Dicision")
    # plt.bar(rand,height=21, label="Rand_For")
    # plt.bar(rand,height=21, label="Rand_For")
    # plt.bar(boosting,height=21, label="Boost")
    # plt.bar(boosting,height=21, label="Boost")
    # plt.bar(bagging,height=21, label="Bagging")
    # plt.bar(bagging,height=21, label="Bagging")
    # plt.bar(stacker,height=21, label="Stacking")
    plt.xticks(bar_width, X_axis)
    plt.legend()
    plt.xlabel("Prediction Model")
    plt.ylabel("No. of Predictions")
    # plt.show()
    plt.savefig('output_graph/final_plot.png')