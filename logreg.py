import math
import matplotlib.pyplot as plt
import random

from sklearn import metrics

# https://en.wikipedia.org/wiki/Activation_function

def sigmoid(x):
    return 1. / (1. + (2.7182818284590452 ** (-x)))


def sigmoid2(x):
    return 1 / (1 + math.pow(math.e, -x))


def sigmoid3(x):
  return 1 / (1 + math.exp(-x))


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


def plot_sigmoid(m):
    SX = []
    SY = []

    for n in list(range(0, 11)):
        np = n * .1

        SX.append(np)
        SY.append(sigmoid(m * np))

    return (SX, SY)


def main():
    train = gen_fake_data()
    test = gen_fake_data(observations=20)

    m = 1.
    learning_rate = 0.1

    X, Y = zip(*train)

    # plt.scatter(X, Y, c=Y, cmap='rainbow')

    # SX, SY = plot_sigmoid(m)
    # plt.plot(SX, SY, color='blue')

    previous_error = 0.

    JX = []
    JY = []

    for epoch in range(10):
        print(f'epoch {epoch}')

        sum_error = 0.
        squared_error = 0.
        predictions = []
        actuals = []

        for t in train:
            x = t[0]
            predicted = sigmoid(m * x)

            actual = t[1]

            delta_m = abs(predicted - actual)
            sum_error += delta_m
            squared_error += (actual - predicted) ** 2

            predictions.append(predicted)
            actuals.append(x)

            # print(predicted, actual, x, delta_m, m)

        mae1 = sum_error / len(train)
        mae2 = (1. / len(train)) * sum_error

        mse1 = squared_error / len(train)
        mse2 = metrics.mean_squared_error(actuals, predictions)

        print('mean abs err', mae1, mae2)
        print('mean sq err', mse1, mse2)

        m -= learning_rate * (previous_error - mae1)

        print('new m', m, learning_rate, previous_error, mae1)
        JX.append(mae1)
        JY.append(m)

        previous_error = mae1

        num_correct = 0.

        for t in test:
            predicted = sigmoid(m * t[0])

            if predicted == t[1]:
                num_correct += 1.

        print('accuracy', num_correct / len(test), m)

        # SX, SY = plot_sigmoid(m)
        # plt.plot(SX, SY, color='green')

    plt.plot(JX, JY, color='purple')

    plt.savefig('file.png')

main()