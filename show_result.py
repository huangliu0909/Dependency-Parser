import matplotlib.pyplot as plt
if __name__ == "__main__":
    dev = {}
    with open("Model_no_dep/dev_result.txt", "r") as f:
        for line in f:
            s = line.replace("\n", "").replace("[", "").replace("]", "").replace("'", "").split(",")
            dev[int(s[0])] = float(s[1])
    print(dev)

    train = {}
    with open("Model_no_dep/train_result.txt", "r") as f:
        for line in f:
            s = line.replace("\n", "").replace("[", "").replace("]", "").replace("'", "").split(",")
            train[int(s[0])] = float(s[1])
    print(train)

    x = [0]
    dev_y = [dev[0]]
    train_y = [train[0]]
    for i in range(154):
        x.append((i+1)*1000)
        dev_y.append(dev[(i+1)*1000])
        train_y.append(train[(i+1)*1000])
    print(x)
    print(dev_y)
    print(train_y)

    plt.figure()
    plt.plot(x, dev_y, label="dev-accuracy", color="red")
    plt.plot(x, train_y, label="train-loss", color="orange")
    plt.xlabel("step(s)")
    plt.legend()
    plt.show()
