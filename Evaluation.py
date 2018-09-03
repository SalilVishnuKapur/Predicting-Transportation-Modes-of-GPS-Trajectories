import pandas as pd
from Classifiers import *
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class Evaluation:
    # This is the implementation of the predict method where you pass your learnt model and it gives you the predicted labels.
    def predictHierarchy(testData, modelDic):
        testData1 = testData.copy()
        indexList = []
        predList = []
        frames = []
        predLabels = []
        pred = modelDic['C1'].predict(testData1)
        testData1['newLabels'] = pred
        grp1 = testData1.groupby('newLabels')
        for grp in grp1:
            if (grp[0] == -100):
                testData2 = grp[1].iloc[:, 0:20]
                pred2 = modelDic['C2'].predict(testData2)
                testData2['newLabels'] = pred2
                grp2 = testData2.groupby('newLabels')
                for grp in grp2:
                    # print('grp2 ->'+ str(grp[0]))
                    if (grp[0] == 80):
                        testData3 = grp[1].iloc[:, 0:20]
                        pred3 = modelDic['C3'].predict(testData3)
                        testData3['newLabels'] = pred3
                        grp3 = testData3.groupby('newLabels')
                        for grp in grp3:
                            # print('grp3 ->'+ str(grp[0]))
                            if (grp[0] == 60):
                                testData4 = grp[1].iloc[:, 0:20]
                                pred4 = modelDic['C4'].predict(testData4)
                                testData4['newLabels'] = pred4
                                grp4 = testData4.groupby('newLabels')
                                for grp in grp4:
                                    # print('grp4 ->'+ str(grp[0]))
                                    if (grp[0] == 40):
                                        testData5 = grp[1].iloc[:, 0:20]
                                        pred5 = modelDic['C5'].predict(testData5)
                                        testData5['newLabels'] = pred5
                                        grp5 = testData5.groupby('newLabels')
                                        for grp in grp5:
                                            # print('grp5 ->'+ str(grp[0]))
                                            if (grp[0] == 20):
                                                predList.append(grp[1].iloc[:, 20])
                                            if (grp[0] == -20):
                                                predList.append(grp[1].iloc[:, 20])
                                    if (grp[0] == -40):
                                        predList.append(grp[1].iloc[:, 20])
                            if (grp[0] == -60):
                                predList.append(grp[1].iloc[:, 20])
                    if (grp[0] == -80):
                        predList.append(grp[1].iloc[:, 20])
            if (grp[0] == 100):
                predList.append(grp[1].iloc[:, 20])
                # Converting the predictions numberical value to corresponding class value i.e {100 -> 'train', -80 -> 'subway',
        # -60 -> 'walk', -40 -> 'car', -20 -> 'taxi', 20 -> 'bus'}if output of hierarchy was 100 then get 'train'. Similarliy
        # for other numerical values to their respective classes.
        for i in range(len(predList)):
            frames.append(pd.DataFrame(predList[i]))
        result = pd.concat(frames)
        predictions = result.sort_index(axis=0, ascending=True)
        for i in predictions['newLabels']:
            if (i == 100):
                predLabels.append('train')
            if (i == -80):
                predLabels.append('subway')
            elif (i == -60):
                predLabels.append('walk')
            elif (i == -40):
                predLabels.append('car')
            elif (i == -20):
                predLabels.append('taxi')
            elif (i == 20):
                predLabels.append('bus')
        return (predLabels)

    def classwiseAccuracy(actual, pred):
        kk = {}
        actual = list(actual)
        keys = list(Counter(actual).keys())  # equals to list(set(words))
        values = list(Counter(actual).values())  # counts the elements' frequency

        for i, j in zip(keys, values):
            score = [1 for word, predWord in zip(actual, pred) if (word == predWord and word == i)]
            kk[i] = sum(score) / j
        return kk

    def cvStratified(trainData, trainLabels, typeOfClassification):
        if (typeOfClassification == 'RandomForestHierarchy'):
            cvRfHierarchyPerClass = []
            cvRfHierarchy = []
            skf = StratifiedKFold(n_splits=10)
            skf.get_n_splits(trainData, trainLabels)
            for train_index, test_index in skf.split(trainData, trainLabels):
                trainIndex = train_index.tolist()
                testIndex = test_index.tolist()
                result = Classifiers.fitHierarchyRFC(trainData.iloc[trainIndex], trainLabels.iloc[trainIndex], {})
                predLabels = Evaluation.predictHierarchy(trainData.iloc[testIndex], result)
                cvRfHierarchy.append(accuracy_score(trainLabels.iloc[testIndex], predLabels))
                cvRfHierarchyPerClass.append(Evaluation.classwiseAccuracy(trainLabels.iloc[testIndex], predLabels))
            return (cvRfHierarchyPerClass, cvRfHierarchy)
        if (typeOfClassification == 'DecisionTreeHierarchy'):
            cvDtHierarchyPerClass = []
            cvDtHierarchy = []
            skf = StratifiedKFold(n_splits=10)
            skf.get_n_splits(trainData, trainLabels)
            for train_index, test_index in skf.split(trainData, trainLabels):
                trainIndex = train_index.tolist()
                testIndex = test_index.tolist()
                result = Classifiers.fitHierarchyDTC(trainData.iloc[trainIndex], trainLabels.iloc[trainIndex], {})
                predLabels = Evaluation.predictHierarchy(trainData.iloc[testIndex], result)
                cvDtHierarchy.append(accuracy_score(trainLabels.iloc[testIndex], predLabels))
                cvDtHierarchyPerClass.append(Evaluation.classwiseAccuracy(trainLabels.iloc[testIndex], predLabels))
            return (cvDtHierarchyPerClass, cvDtHierarchy)
        if (typeOfClassification == 'RandomForestFlat'):
            cvRfFlatPerClass = []
            cvRfFlat = []
            skf = StratifiedKFold(n_splits=10)
            skf.get_n_splits(trainData, trainLabels)
            for train_index, test_index in skf.split(trainData, trainLabels):
                trainIndex = train_index.tolist()
                testIndex = test_index.tolist()
                rfc = RandomForestClassifier()
                rfc.fit(trainData.iloc[trainIndex], trainLabels.iloc[trainIndex])
                predFlatRFC = rfc.predict(trainData.iloc[testIndex])
                cvRfFlat.append(accuracy_score(trainLabels.iloc[testIndex], predFlatRFC))
                cvRfFlatPerClass.append(Evaluation.classwiseAccuracy(trainLabels.iloc[testIndex], predFlatRFC))
            return (cvRfFlatPerClass, cvRfFlat)
        if (typeOfClassification == 'DecisionTreeFlat'):
            cvDtFlatPerClass = []
            cvDtFlat = []
            skf = StratifiedKFold(n_splits=10)
            skf.get_n_splits(trainData, trainLabels)
            for train_index, test_index in skf.split(trainData, trainLabels):
                trainIndex = train_index.tolist()
                testIndex = test_index.tolist()
                dtc = DecisionTreeClassifier()
                dtc.fit(trainData.iloc[trainIndex], trainLabels.iloc[trainIndex])
                predFlatDTC = dtc.predict(trainData.iloc[testIndex])
                cvDtFlat.append(accuracy_score(trainLabels.iloc[testIndex], predFlatDTC))
                cvDtFlatPerClass.append(Evaluation.classwiseAccuracy(trainLabels.iloc[testIndex], predFlatDTC))
            return (cvDtFlatPerClass, cvDtFlat)

