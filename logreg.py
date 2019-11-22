import math
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    return 1. / (1. + (2.7182818284590452 ** (-x)))


def output(m, x, b):
    return (m * x) + b


def cross_entropy(predicted, actual):
    L = ((-actual) * math.log(predicted))
    R = ((-actual) * math.log(-predicted))
    return  L - R


def gen_fake_data(observations=100, intercept=.35):
    d = []

    for n in range(observations):
        r = random.random()
        d.append((r, 1 if r >= intercept else 0))

    return d


def plot_sigmoid(m, b):
    SX = []
    SY = []

    for n in list(range(-10, 10)):
        np = n * .1

        SX.append(n)
        SY.append(sigmoid(output(m, np, b)))

    return (SX, SY)


def main():
    train = gen_fake_data()
    test = gen_fake_data(observations=20)

    m = 1.
    b = 1.
    learning_rate = 0.01

    X, Y = zip(*train)

    plt.scatter(X, Y, c=Y, cmap='rainbow')

    SX, SY = plot_sigmoid(m, b)
    plt.plot(SX, SY, color='blue')

    for epoch in range(2000):
        print(f'epoch {epoch}')

        for t in train:
            x = t[0]
            predicted = sigmoid(output(m, x, b))
            actual = t[1]

            delta_m = x * (predicted - actual)
            print(predicted, actual, x, delta_m, m)

            m = m - (delta_m * learning_rate)

            # print(predicted, actual, cost, m, b, delta_m, delta_b)

        num_correct = 0.

        for t in test:
            predicted = sigmoid(output(m, x, b))

            if predicted == t[1]:
                num_correct += 1.

        print('accuracy', num_correct / len(test), m, b)

    SX, SY = plot_sigmoid(m, b)
    plt.plot(SX, SY, color='green')

    plt.savefig('file.png')

main()