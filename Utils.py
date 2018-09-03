



class Utils:

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