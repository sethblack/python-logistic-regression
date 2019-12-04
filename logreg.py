import math
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    return 1. / (1. + (2.7182818284590452 ** (-x)))


def sigmoid2(x):
    return 1 / (1 + math.pow(math.e, -x))


def sigmoid3(x):
  return 1 / (1 + math.exp(-x))


def output(m, x, b):
    return (m * x) + b


def cross_entropy(predicted, actual):
    L = ((-actual) * math.log(predicted))
    R = ((-actual) * math.log(-predicted))
    return  L - R


def gen_fake_data(observations=100, intercept=.35, fuzz=.06):
    def fuzzit(i):
        return i + (random.randint(fuzz * -100, fuzz * 100) / 100.)

    d = []

    for n in range(observations):
        r = random.random()
        d.append((r, 1 if r >= fuzzit(intercept) else 0))

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

    # SX, SY = plot_sigmoid(m, b)
    # plt.plot(SX, SY, color='blue')

    for epoch in range(2):
        print(f'epoch {epoch}')
        sum_error = 0

        for t in train:
            x = t[0]
            predicted = sigmoid(output(m, x, b))

            actual = t[1]

            delta_m = abs(predicted - actual)
            sum_error += delta_m

            print(predicted, actual, x, delta_m, m, b)

        mae1 = sum_error / len(train)
        mae2 = (1. / len(train)) * sum_error
        print('mean abs err', mae1, mae2)

        # num_correct = 0.

        # for t in test:
        #     predicted = sigmoid(output(m, t[0], b))

        #     if predicted == t[1]:
        #         num_correct += 1.

        # print('accuracy', num_correct / len(test), m, b)

    # SX, SY = plot_sigmoid(m, b)
    # plt.plot(SX, SY, color='green')

    plt.savefig('file.png')

main()