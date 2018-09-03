from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from Utils import *


class Classifiers:
    # This is the implementation of the proposed hierarchy above using Random Forest Classifier
    # This hierarchy learns on the bases of the relabling method above
    def fitHierarchyRFC(trainData, trainLabels, modelDic):
        trainData1 = trainData.copy()
        label = Utils.relabel(1, trainLabels)
        C1 = RandomForestClassifier().fit(trainData1, label)
        modelDic['C1'] = C1
        trainData1['oldLabels'] = trainLabels
        trainData1['newLabels'] = label
        grp1 = trainData1.groupby('newLabels')
        for grp in grp1:
            if (grp[0] == -100):
                trainData2 = grp[1].iloc[:, 0:20]
                trainLabels2 = grp[1]['oldLabels']
                labels2 = Utils.relabel(2, trainLabels2)
                C2 = RandomForestClassifier().fit(trainData2, labels2)
                modelDic['C2'] = C2
                trainData2['oldLabels'] = trainLabels2
                trainData2['newLabels'] = labels2
                grp2 = trainData2.groupby('newLabels')
                for grp in grp2:
                    if (grp[0] == 80):
                        trainData3 = grp[1].iloc[:, 0:20]
                        trainLabels3 = grp[1]['oldLabels']
                        labels3 = Utils.relabel(3, trainLabels3)
                        C3 = RandomForestClassifier().fit(trainData3, labels3)
                        modelDic['C3'] = C3
                        trainData3['oldLabels'] = trainLabels3
                        trainData3['newLabels'] = labels3
                        grp3 = trainData3.groupby('newLabels')
                        for grp in grp3:
                            if (grp[0] == 60):
                                trainData4 = grp[1].iloc[:, 0:20]
                                trainLabels4 = grp[1]['oldLabels']
                                labels4 = Utils.relabel(4, trainLabels4)
                                C4 = RandomForestClassifier().fit(trainData4, labels4)
                                modelDic['C4'] = C4
                                trainData4['oldLabels'] = trainLabels4
                                trainData4['newLabels'] = labels4
                                grp4 = trainData4.groupby('newLabels')
                                for grp in grp4:
                                    if (grp[0] == 40):
                                        trainData5 = grp[1].iloc[:, 0:20]
                                        trainLabels5 = grp[1]['oldLabels']
                                        labels5 = Utils.relabel(5, trainLabels5)
                                        C5 = RandomForestClassifier().fit(trainData5, labels5)
                                        modelDic['C5'] = C5
        return modelDic

    # This is the implementation of the proposed hierarchy above using Decision Tree Classifier
    # This hierarchy learns on the bases of the relabling method above
    def fitHierarchyDTC(trainData, trainLabels, modelDic):
        trainData1 = trainData.copy()
        label = Utils.relabel(1, trainLabels)
        C1 = DecisionTreeClassifier().fit(trainData1, label)
        modelDic['C1'] = C1
        trainData1['oldLabels'] = trainLabels
        trainData1['newLabels'] = label
        grp1 = trainData1.groupby('newLabels')
        for grp in grp1:
            if (grp[0] == -100):
                trainData2 = grp[1].iloc[:, 0:20]
                trainLabels2 = grp[1]['oldLabels']
                labels2 = Utils.relabel(2, trainLabels2)
                C2 = DecisionTreeClassifier().fit(trainData2, labels2)
                modelDic['C2'] = C2
                trainData2['oldLabels'] = trainLabels2
                trainData2['newLabels'] = labels2
                grp2 = trainData2.groupby('newLabels')
                for grp in grp2:
                    if (grp[0] == 80):
                        trainData3 = grp[1].iloc[:, 0:20]
                        trainLabels3 = grp[1]['oldLabels']
                        labels3 = Utils.relabel(3, trainLabels3)
                        C3 = DecisionTreeClassifier().fit(trainData3, labels3)
                        modelDic['C3'] = C3
                        trainData3['oldLabels'] = trainLabels3
                        trainData3['newLabels'] = labels3
                        grp3 = trainData3.groupby('newLabels')
                        for grp in grp3:
                            if (grp[0] == 60):
                                trainData4 = grp[1].iloc[:, 0:20]
                                trainLabels4 = grp[1]['oldLabels']
                                labels4 = Utils.relabel(4, trainLabels4)
                                C4 = DecisionTreeClassifier().fit(trainData4, labels4)
                                modelDic['C4'] = C4
                                trainData4['oldLabels'] = trainLabels4
                                trainData4['newLabels'] = labels4
                                grp4 = trainData4.groupby('newLabels')
                                for grp in grp4:
                                    if (grp[0] == 40):
                                        trainData5 = grp[1].iloc[:, 0:20]
                                        trainLabels5 = grp[1]['oldLabels']
                                        labels5 = Utils.relabel(5, trainLabels5)
                                        C5 = DecisionTreeClassifier().fit(trainData5, labels5)
                                        modelDic['C5'] = C5
        return modelDic

