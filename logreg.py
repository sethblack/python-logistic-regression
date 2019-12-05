import math
import matplotlib.pyplot as plt
import random

from sklearn import metrics

# https://en.wikipedia.org/wiki/Activation_function

def sigmoid(x):
    return 1 / (1 + (2.7182818284590452 ** (-x)))


def sigmoid2(x):
    return 1 / (1 + math.pow(math.e, -x))


def sigmoid3(x):
  return 1 / (1 + math.exp(-x))


def cross_entropy(actuals, predictions):
    m = len(predictions)

    ce = 0.

    for Y, Y_hat in zip(actuals, predictions):
        Y_hat = Y_hat if Y_hat < 1 else 1 - 1e-15

        ce += ((Y * math.log(Y_hat)) + ((1 - Y) * (math.log(1 - Y_hat))))

    return (-1 / m) * ce


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

    m = -100.
    learning_rate = 0.1

    X, Y = zip(*train)

    # plt.scatter(X, Y, c=Y, cmap='rainbow')

    # SX, SY = plot_sigmoid(m)
    # plt.plot(SX, SY, color='blue')

    previous_error = 0.

    JX = []
    JY = []

    for epoch in range(2000):
        print(f'epoch {epoch}')

        sum_error = 0.
        squared_error = 0.
        predictions = []
        actuals = []

        for t in train:
            x = t[0]
            predicted = sigmoid3(m * x)

            actual = t[1]

            delta_m = abs(actual - predicted)
            sum_error += delta_m
            squared_error += (actual - predicted) ** 2

            predictions.append(predicted)
            actuals.append(actual)

            # print(predicted, actual, x, delta_m, m)

        mae1 = (1. / len(train)) * sum_error
        mae2 = metrics.mean_absolute_error(actuals, predictions)

        mse1 = (1. / len(train)) * squared_error
        mse2 = metrics.mean_squared_error(actuals, predictions)

        ce1 = cross_entropy(actuals, predictions)
        ce2 = metrics.log_loss(actuals, predictions)

        print('m:', m, 'mse:', mse1, 'ce1:', ce1, 'ce2', ce2)

        JX.append(m)
        JY.append(ce2)

        #m -= learning_rate * mse1
        m += learning_rate * ce2

        # print('updated m:', m, 'learning rate:', learning_rate, 'previous err:', previous_error)

        num_correct = 0.

        for t in test:
            predicted = int(sigmoid3(m * t[0]))

            if predicted == t[1]:
                num_correct += 1.

            print('p', predicted, 't', t[1])

        print('accuracy', num_correct / len(test), m)

        # SX, SY = plot_sigmoid(m)
        # plt.plot(SX, SY, color='green')

    plt.scatter(JX, JY, color='purple')

    plt.savefig('file.png')

main()