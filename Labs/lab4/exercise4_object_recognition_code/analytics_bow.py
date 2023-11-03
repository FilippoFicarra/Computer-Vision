import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open("output.txt") as f:
        lines = f.readlines()
        
    pos_acc = []
    neg_acc = []
    for line in lines:
        if "test pos sample accuracy" in line:
            pos_acc.append(float(line.split(":")[1].lstrip()))
        elif "test neg sample accuracy" in line:
            neg_acc.append(float(line.split(":")[1].lstrip()))
            

    plt.plot(np.arange(1, len(pos_acc)+1), pos_acc, label="pos")
    plt.plot(np.arange(1, len(neg_acc)+1), neg_acc, label="neg")
    plt.legend()
    plt.xlabel("runs")
    plt.ylabel("accuracy")
    plt.savefig("pos_neg.png")