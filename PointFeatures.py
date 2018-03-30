from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
from operator import mul
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from haversine import haversine
from datetime import datetime
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Loading the Input Data and sorting it in ascending order of 't_user_id' and 'collected_time'. This way of sorting
# the data is equivalent to grouping the data on the bases of 't_user_id' and 'collected_time'.
df = pd.read_csv('geolife_raw.csv')
total = len(df)
df = df.sort_values(['t_user_id', 'collected_time'], ascending=True).reset_index(drop=True)

# Preprocessing
x = df['collected_time'].str.split(' ', expand = True)
df['date_Start'] = x[0]
y = x[1].str.split('-', expand = True)
df['time_Start'] = y[0]
df['latitude_Start'] = df['latitude']
df['longitude_Start'] = df['longitude']

df['latitude_End'] = df['latitude'].drop([0]).reset_index(drop=True)
df['longitude_End'] = df['longitude'].drop([0]).reset_index(drop=True)

df['date_End'] = df['date_Start'].drop([0]).reset_index(drop=True)
df['time_End'] = df['time_Start'].drop([0]).reset_index(drop=True)

df['UserChk'] = df['t_user_id'].drop([0]).reset_index(drop=True)
df['ModeChk'] = df['transportation_mode'].drop([0]).reset_index(drop=True)
df = df.drop('collected_time', axis =1)
df = df.drop('latitude', axis =1)
df = df.drop('longitude', axis =1)
df = df.drop([total-1], axis =0)

# The above preprocessing has created a DataFrame with the columns arranged in a much more better computational way.
# Columns of processable data now are :- 
# 't_user_id', 'transportation_mode', 'date_Start', 'time_Start', 'latitude_Start', 'longitude_Start',
# 'latitude_End', 'longitude_End','date_End', 'time_End', 'UserChk', 'ModeChk' 
# We finally convert this DataFrame to list of lists because of the better time complexity of lists than pandas DataFrame
dataList = df.values.tolist()

# This is the method to calculate bearing between two points on the bases
# of latitute and longitutes of the 2 points.
def bearing_Calculator(row):
    start, end = ((row[4], row[5]), (row[6], row[7])) 
    lat1 = math.radians(float(start[0]))
    lat2 = math.radians(float(end[0]))

    diffLong = math.radians(float(end[1]) - float(start[1]))
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)* math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

# This below method will create this kind of list of list structure
#         s -> (s0,s1), (s1,s2), (s2, s3), ...
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.zip_longest(a, b)

# Here we are calculating distance, speed, acceleration and bearing.

# Filtering the data so as to remove as per our preprocessed data and understanding of trajectories.
# We are filtering:-
# 1. The information if starting inf is from 1 user and ending inf is from another user
# 2. The information if starting inf is from 1 transportation mode and ending inf is from another transportation mode
# 3. If the starting date and ending date match or not
FMT = '%H:%M:%S'
filteredData = [item for item in dataList if(item[0]==item[10] and item[1]==item[11] and item[2]==item[8])]

# Here we are creating a flag numerical column so as to easily find when there is a change in subtrajectory or trajectory
startId = filteredData[0][0]
startMode = filteredData[0][1]
startDate = filteredData[0][2]
subTrajGrper = []
count = 1
for row in filteredData:
    if(startId ==row[0] and startMode ==row[1] and startDate ==row[2]):
        subTrajGrper.append(count)
    else:
        startId = row[0]
        startMode = row[1]
        startDate = row[2]
        count+=1
        subTrajGrper.append(count)

# Calculating Distance
distance = [haversine((float(row[4]),float(row[5])), (float(row[6]),float(row[7])))*1000.0 for row in filteredData]
# Calculating Time
time = [(datetime.strptime(str(row[9]), FMT) - datetime.strptime(str(row[3]), FMT)).seconds for row in filteredData]
# Calculating speed
speed = [x/y if y != 0 else 0 for x,y in zip(distance, time)]
# Calculating acceleration
pairedSpeed = list(pairwise(speed))
acceleration = [(x[1]-x[0])/y if(y != 0 and x[1]!=None) else 0 for x,y in zip(pairedSpeed, time) ]
# Calculating Bearing
bearing = [bearing_Calculator(row) for row in filteredData]

# Here we are doing a list compression so as to add the answer of Q1 to our preprocessed data. 
dataA1Soln = [u + [v,w,x,y,z]  for u,v,w,x,y,z in zip(filteredData,subTrajGrper, distance, speed, acceleration, bearing)]

# Here we are masking the accleration to 0 in case it is calculated by change in speed between 2 different users.
pairedA1 = list(pairwise(dataA1Soln))
dataA1Soln = [list(map(mul, rows[0], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1])) if(rows[1]!= None and rows[0][12] != rows[1][12]) else rows[0] for rows in pairedA1]

##### Creating sub trajectories #####

# We are filtering the data to contain only the useful columns for calculating A2 and A3
list1 = [0,1,2,12,13,14,15,16]
dataImp = [[each_list[i] for i in list1] for each_list in dataA1Soln]

# Grouping the data for A2
dataSubTrajectory = [list(items) for _, items in groupby(dataImp, itemgetter(0,1,2,3))]

# Filtering the subtrajectories which have points less than 10 
dataFiltSubTrj = [grp for grp in dataSubTrajectory if(len(grp)>10)]


# A method for calculating all the statistical values asked in A2
def stats_Calculator(data):        
        mini = np.min(data)
        maxi = np.max(data)
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        return [mini, maxi, mean, median,std]


# Calculating all the statistical values for A2. Here we calculate the minimum, maximum, mean and median
# for every subtrajectory.
A2Traj = []
for grp in dataFiltSubTrj: 
        count+=1
        statsDistance = stats_Calculator([distanceRow[4] for distanceRow in grp])
        statsSpeed = stats_Calculator([speedRow[5] for speedRow in grp])
        statsAcceleration = stats_Calculator([accRow[6] for accRow in grp])
        statsBearing = stats_Calculator([bearRow[7] for bearRow in grp])
      
        x1 = [grp[0][0], grp[0][1], grp[0][2], grp[0][3]]
        A2Traj.append(x1+statsDistance+statsSpeed+statsAcceleration+statsBearing)


# Filtering the subrajectories of motorcycle and run
A2FiltTraj = [trj for trj in A2Traj if(trj[1]!='motorcycle' and trj[1]!='run') ]

# Filtering the data so as to keep only those columns which will be useful for analysing the feature values by class. 
list2 = [0,1,2,3,6,11,16,21]
A2FiltTrajF = [[each_list2[i] for i in list2] for each_list2 in A2FiltTraj]

# Convert the filtered data into pandas DataFrame as it has a very optimized groupby function
output = pd.DataFrame(A2FiltTrajF,columns = ['t_user_id', 'transportation_mode', 'date_Start', 'Flag' 
                               , 'meanDis', 'meanSpeed', 'meanAcc', 'meanBrng'])

# Grouping by the mode so as to analyse the silimarities and disimilarities between classes 
outgrp = output.groupby(['transportation_mode'])

# Computing the mean per class for the 4 feature values i.e distance, speed, acceleration and bearing.
dicPerType = {}
for grpType in outgrp:
    label = grpType[0]
    grp = grpType[1]
    data= []
    data.append(np.mean(grp['meanDis']))
    data.append(np.mean(grp['meanSpeed']))
    data.append(np.mean(grp['meanAcc']))
    data.append(np.mean(grp['meanBrng']))
    dicPerType[label] = data
    
# Plotting analysis using bar plot
count = 0
features = [0, 1, 2, 3]
keys = ['mean distance', 'mean speed', 'mean acceleration', 'mean bearing']
xLabels = ['bus', 'car', 'subway', 'taxi', 'train', 'walk']
for subset in range(4):
        plt.subplot(int(str(22) +''+ str(count+1)))
        x = range(6)
        print(dicPerType['bus'][subset], dicPerType['car'][subset], dicPerType['subway'][subset], 
             dicPerType['taxi'][subset], dicPerType['train'][subset], dicPerType['walk'][subset])
        width = 1/1.5
        plt.bar(x, list([dicPerType['bus'][subset], dicPerType['car'][subset], dicPerType['subway'][subset], 
             dicPerType['taxi'][subset], dicPerType['train'][subset], dicPerType['walk'][subset]]), width, color="blue")
        plt.xlabel('6 Classes')
        plt.ylabel(keys[count])
        plt.xticks(range(len(xLabels)), xLabels, size='small')
        plt.subplots_adjust( hspace= 0.5 )
        count+=1
plt.show()

# Cleaning data by removing unimportant columns and creating a DataFrame for classification process. 

dataSubTrajectories = pd.DataFrame(A2FiltTraj, columns = ['t_user_id', 'transportation_mode', 'date_Start', 'flag' 
                               , 'minDis' ,'maxDis', 'meanDis', 'medianDis', 'stdDis'
                               , 'minSpeed' ,'maxSpeed', 'meanSpeed', 'medianSpeed', 'stdSpeed'
                               , 'minAcc' ,'maxAcc', 'meanAcc', 'medianAcc', 'stdAcc'
                              , 'minBrng' ,'maxBrng', 'meanBrng', 'medianBrng', 'stdBrng']  )

#dataSubTrajectories = pd.read_csv('dataFinal_A1.txt', delimiter = '\t')
dataSubTrajectories = dataSubTrajectories.drop('t_user_id', axis =1)
dataSubTrajectories = dataSubTrajectories.drop('date_Start', axis =1)
dataSubTrajectories = dataSubTrajectories.drop('flag', axis =1)

# This is the relabling method which is used in the hierarchical structure
# Coded on the bases of Q1 of this section
def relabel(node, labels):
    lb = []
    if(node == 1):
        for value in labels:
            if(value=='train'):
                lb.append(100)
            else:
                lb.append(-100)    
    elif(node == 2):
        for value in labels:
            if(value=='subway'):
                lb.append(-80)
            else:
                lb.append(80)
    elif(node == 3):
        for value in labels:
            if(value=='walk'):
                lb.append(-60)
            else:
                lb.append(60)
    elif(node == 4):
        for value in labels:
            if(value=='car'):
                lb.append(-40)
            else:
                lb.append(40)
    elif(node == 5):
        for value in labels:
            if(value=='taxi'):
                lb.append(-20)
            else:
                lb.append(20)
    return lb

# This is the implementation of the proposed hierarchy above using Random Forest Classifier
# This hierarchy learns on the bases of the relabling method above  
def fitHierarchyRFC(trainData,trainLabels, modelDic):
    trainData1 = trainData.copy()
    label = relabel(1,trainLabels)   
    C1 = RandomForestClassifier().fit(trainData1, label)
    modelDic['C1'] = C1
    trainData1['oldLabels'] = trainLabels
    trainData1['newLabels'] = label
    grp1 = trainData1.groupby('newLabels')
    for grp in grp1: 
        if(grp[0] == -100):
            trainData2 = grp[1].iloc[:,0:20]
            trainLabels2 = grp[1]['oldLabels']
            labels2 = relabel(2,trainLabels2)
            C2 = RandomForestClassifier().fit(trainData2, labels2)
            modelDic['C2'] = C2
            trainData2['oldLabels'] = trainLabels2
            trainData2['newLabels'] = labels2
            grp2 = trainData2.groupby('newLabels')
            for grp in grp2:
                if(grp[0] == 80):
                    trainData3 = grp[1].iloc[:,0:20]
                    trainLabels3 = grp[1]['oldLabels']
                    labels3 = relabel(3,trainLabels3)
                    C3 = RandomForestClassifier().fit(trainData3, labels3)
                    modelDic['C3'] = C3
                    trainData3['oldLabels'] = trainLabels3
                    trainData3['newLabels'] = labels3
                    grp3 = trainData3.groupby('newLabels')
                    for grp in grp3:
                        if(grp[0] == 60):
                            trainData4 = grp[1].iloc[:,0:20]
                            trainLabels4 = grp[1]['oldLabels']
                            labels4 = relabel(4,trainLabels4)
                            C4 = RandomForestClassifier().fit(trainData4, labels4)
                            modelDic['C4'] = C4
                            trainData4['oldLabels'] = trainLabels4
                            trainData4['newLabels'] = labels4
                            grp4 = trainData4.groupby('newLabels')
                            for grp in grp4:
                                if(grp[0] == 40):
                                    trainData5 = grp[1].iloc[:,0:20]
                                    trainLabels5 = grp[1]['oldLabels']
                                    labels5 = relabel(5,trainLabels5)
                                    C5 = RandomForestClassifier().fit(trainData5, labels5)
                                    modelDic['C5'] = C5
    return modelDic

# This is the implementation of the proposed hierarchy above using Decision Tree Classifier
# This hierarchy learns on the bases of the relabling method above  
def fitHierarchyDTC(trainData,trainLabels, modelDic):
    trainData1 = trainData.copy()
    label = relabel(1,trainLabels)   
    C1 = DecisionTreeClassifier().fit(trainData1, label)
    modelDic['C1'] = C1
    trainData1['oldLabels'] = trainLabels
    trainData1['newLabels'] = label
    grp1 = trainData1.groupby('newLabels')
    for grp in grp1: 
        if(grp[0] == -100):
            trainData2 = grp[1].iloc[:,0:20]
            trainLabels2 = grp[1]['oldLabels']
            labels2 = relabel(2,trainLabels2)
            C2 = DecisionTreeClassifier().fit(trainData2, labels2)
            modelDic['C2'] = C2
            trainData2['oldLabels'] = trainLabels2
            trainData2['newLabels'] = labels2
            grp2 = trainData2.groupby('newLabels')
            for grp in grp2:
                if(grp[0] == 80):
                    trainData3 = grp[1].iloc[:,0:20]
                    trainLabels3 = grp[1]['oldLabels']
                    labels3 = relabel(3,trainLabels3)
                    C3 = DecisionTreeClassifier().fit(trainData3, labels3)
                    modelDic['C3'] = C3
                    trainData3['oldLabels'] = trainLabels3
                    trainData3['newLabels'] = labels3
                    grp3 = trainData3.groupby('newLabels')
                    for grp in grp3:
                        if(grp[0] == 60):
                            trainData4 = grp[1].iloc[:,0:20]
                            trainLabels4 = grp[1]['oldLabels']
                            labels4 = relabel(4,trainLabels4)
                            C4 = DecisionTreeClassifier().fit(trainData4, labels4)
                            modelDic['C4'] = C4
                            trainData4['oldLabels'] = trainLabels4
                            trainData4['newLabels'] = labels4
                            grp4 = trainData4.groupby('newLabels')
                            for grp in grp4:
                                if(grp[0] == 40):
                                    trainData5 = grp[1].iloc[:,0:20]
                                    trainLabels5 = grp[1]['oldLabels']
                                    labels5 = relabel(5,trainLabels5)
                                    C5 = DecisionTreeClassifier().fit(trainData5, labels5)
                                    modelDic['C5'] = C5
    return modelDic


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
        if(grp[0] == -100):
            testData2 = grp[1].iloc[:,0:20]
            pred2 = modelDic['C2'].predict(testData2)
            testData2['newLabels'] = pred2
            grp2 = testData2.groupby('newLabels')
            for grp in grp2:
                #print('grp2 ->'+ str(grp[0]))
                if(grp[0] == 80):
                    testData3 = grp[1].iloc[:,0:20]
                    pred3 = modelDic['C3'].predict(testData3)
                    testData3['newLabels'] = pred3
                    grp3 = testData3.groupby('newLabels')
                    for grp in grp3:
                        #print('grp3 ->'+ str(grp[0]))
                        if(grp[0] == 60):
                            testData4 = grp[1].iloc[:,0:20]
                            pred4 = modelDic['C4'].predict(testData4)
                            testData4['newLabels'] = pred4
                            grp4 = testData4.groupby('newLabels')
                            for grp in grp4:
                                #print('grp4 ->'+ str(grp[0]))
                                if(grp[0] == 40):
                                    testData5 = grp[1].iloc[:,0:20]
                                    pred5 = modelDic['C5'].predict(testData5)
                                    testData5['newLabels'] = pred5
                                    grp5 = testData5.groupby('newLabels')
                                    for grp in grp5:
                                        #print('grp5 ->'+ str(grp[0]))
                                        if(grp[0] == 20):
                                            predList.append(grp[1].iloc[:,20])
                                        if(grp[0] == -20):
                                            predList.append(grp[1].iloc[:,20])    
                                if(grp[0] == -40):
                                    predList.append(grp[1].iloc[:,20])    
                        if(grp[0] == -60):
                            predList.append(grp[1].iloc[:,20])    
                if(grp[0] == -80):
                    predList.append(grp[1].iloc[:,20])            
        if(grp[0] == 100):  
            predList.append(grp[1].iloc[:,20]) 
    # Converting the predictions numberical value to corresponding class value i.e {100 -> 'train', -80 -> 'subway',
    # -60 -> 'walk', -40 -> 'car', -20 -> 'taxi', 20 -> 'bus'}if output of hierarchy was 100 then get 'train'. Similarliy
    # for other numerical values to their respective classes.
    for i in range(len(predList)):
        frames.append(pd.DataFrame(predList[i]))
    result = pd.concat(frames)
    predictions = result.sort_index(axis=0, ascending=True)
    for i in predictions['newLabels']:
        if(i==100):
            predLabels.append('train')
        if(i==-80):
            predLabels.append('subway')
        elif(i==-60):
            predLabels.append('walk')
        elif(i==-40):
            predLabels.append('car')
        elif(i==-20):
            predLabels.append('taxi')
        elif(i==20):
            predLabels.append('bus')
    return (predLabels)

modelDic = {}
trainData = dataSubTrajectories.iloc[0:4708,1:21]
trainLabels  = dataSubTrajectories.iloc[0:4708,0]
testData = dataSubTrajectories.iloc[4708:5885,1:21]
testLabels  = dataSubTrajectories.iloc[4708:5885,0]

result = fitHierarchyRFC(trainData, trainLabels, modelDic)
predLabels = predictHierarchy(testData, result)
target_names = ['train', 'subway', 'walk', 'car', 'taxi', 'bus' ]
print("CLASSIFICATION REPORT :- ")
print(classification_report(testLabels, predLabels, target_names=target_names))
print("ACCURACY OF COMPLETE HIERARCHY :- ")
print(accuracy_score(testLabels, predLabels))

rfc = RandomForestClassifier()
rfc.fit(trainData, trainLabels)
predFlatRFC = rfc.predict(testData)
target_names = ['train', 'subway', 'walk', 'car', 'taxi', 'bus' ]
print("CLASSIFICATION REPORT :- ")
print(classification_report(testLabels, predFlatRFC, target_names=target_names))
print("ACCURACY OF COMPLETE FLAT STRUCTURE :- ")
print(accuracy_score(testLabels, predFlatRFC))

result = fitHierarchyDTC(trainData, trainLabels, modelDic)
predLabels = predictHierarchy(testData, result)
target_names = ['train', 'subway', 'walk', 'car', 'taxi', 'bus' ]
print("CLASSIFICATION REPORT :- ")
print(classification_report(testLabels, predLabels, target_names=target_names))
print("ACCURACY OF COMPLETE HIERARCHY :- ")
print(accuracy_score(testLabels, predLabels))

dtc = DecisionTreeClassifier()
dtc.fit(trainData, trainLabels)
predFlatDTC = dtc.predict(testData)
target_names = ['train', 'subway', 'walk', 'car', 'taxi', 'bus' ]
print("CLASSIFICATION REPORT :- ")
print(classification_report(testLabels, predFlatDTC, target_names=target_names))
print("ACCURACY OF COMPLETE FLAT STRUCTURE :- ")
print(accuracy_score(testLabels, predFlatDTC))

           
