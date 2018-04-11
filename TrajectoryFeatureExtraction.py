
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
omes = ('Bus', 'Car ' , 'Subway', 'Taxi', 'Train', 'Walk')
arrRfH = [np.mean(cvRfHierarchyT[0]), np.mean(cvRfHierarchyT[1]), np.mean(cvRfHierarchyT[2]), np.mean(cvRfHierarchyT[3]), np.mean(cvRfHierarchyT[4]), np.mean(cvRfHierarchyT[5])] 
arrRfF =  [np.mean(cvRfFlatT[0]), np.mean(cvRfFlatT[1]), np.mean(cvRfFlatT[2]), np.mean(cvRfFlatT[3]), np.mean(cvRfFlatT[4]), np.mean(cvRfFlatT[5]) ]


N = 6

locx = np.arange(6)  # the x locations for the groups
width = 0.15       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(locx, arrRfH, width, color='r' )
rects2 = ax.bar(locx + width, arrRfF, width, color='y')

ax.set_ylabel('Accuracy Values')
ax.set_title('Class wise RF comparison among Hierarchy and Flat')
ax.set_xticks(locx + width / 2)
ax.set_xticklabels(model_names)
ax.set_yticks(np.arange(0, 1.04, 0.05))
ax.legend((rects1[0], rects2[0]), ('Hierarchy Structure', 'Flat Structure'))


plt.show()utput = pd.DataFrame(A2FiltTrajF,columns = ['t_user_id', 'transportation_mode', 'date_Start', 'Flag' 
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