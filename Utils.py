import math
import itertools
import numpy as np

class Utils:
    '''
    Creating a column for creating DataFrame
    '''
    columns = ['t_user_id', 'transportation_mode', 'date_Start', 'flag'
        , 'minDis', 'maxDis', 'meanDis', 'medianDis', 'stdDis'
        , 'minSpeed', 'maxSpeed', 'meanSpeed', 'medianSpeed', 'stdSpeed'
        , 'minAcc', 'maxAcc', 'meanAcc', 'medianAcc', 'stdAcc'
        , 'minBrng', 'maxBrng', 'meanBrng', 'medianBrng', 'stdBrng']
    # This is the method to calculate bearing between two points on the bases
    # of latitute and longitutes of the 2 points.
    def bearing_Calculator(row):
        start, end = ((row[4], row[5]), (row[6], row[7]))
        lat1 = math.radians(float(start[0]))
        lat2 = math.radians(float(end[0]))

        diffLong = math.radians(float(end[1]) - float(start[1]))
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
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

    # A method for calculating all the statistical values asked in A2
    def stats_Calculator(data):
        mini = np.min(data)
        maxi = np.max(data)
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        return [mini, maxi, mean, median, std]

    def relabel(node, labels):
        lb = []
        if (node == 1):
            for value in labels:
                if (value == 'train'):
                    lb.append(100)
                else:
                    lb.append(-100)
        elif (node == 2):
            for value in labels:
                if (value == 'subway'):
                    lb.append(-80)
                else:
                    lb.append(80)
        elif (node == 3):
            for value in labels:
                if (value == 'walk'):
                    lb.append(-60)
                else:
                    lb.append(60)
        elif (node == 4):
            for value in labels:
                if (value == 'car'):
                    lb.append(-40)
                else:
                    lb.append(40)
        elif (node == 5):
            for value in labels:
                if (value == 'taxi'):
                    lb.append(-20)
                else:
                    lb.append(20)
        return lb

    def transformer(data):
        c1 = []
        c2 = []
        c3 = []
        c4 = []
        c5 = []
        c6 = []
        for rowDic in data:
            c1.append(rowDic['bus'])
            c2.append(rowDic['car'])
            c3.append(rowDic['subway'])
            c4.append(rowDic['taxi'])
            c5.append(rowDic['train'])
            c6.append(rowDic['walk'])
        return [c1, c2, c3, c4, c5, c6]