import matplotlib.pyplot as plt

class Plotter:
    def plotSimilarities(dicPerType, keys, xLabels):
        count = 0
        features = [0, 1, 2, 3]
        keys = ['mean distance', 'mean speed', 'mean acceleration', 'mean bearing']
        xLabels = ['bus', 'car', 'subway', 'taxi', 'train', 'walk']
        for subset in range(4):
            plt.subplot(int(str(22) + '' + str(count + 1)))
            x = range(6)
            print(dicPerType['bus'][subset], dicPerType['car'][subset], dicPerType['subway'][subset],
                  dicPerType['taxi'][subset], dicPerType['train'][subset], dicPerType['walk'][subset])
            width = 1 / 1.5
            plt.bar(x, list([dicPerType['bus'][subset], dicPerType['car'][subset], dicPerType['subway'][subset],
                             dicPerType['taxi'][subset], dicPerType['train'][subset], dicPerType['walk'][subset]]),
                    width, color="blue")
            plt.xlabel('6 Classes')
            plt.ylabel(keys[count])
            plt.xticks(range(len(xLabels)), xLabels, size='small')
            plt.subplots_adjust(hspace=0.5)
            count += 1
        plt.show()