import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


if __name__=='__main__':
    l0 = [Dense(units=1, input_shape=[1])]
    model = Sequential(l0)
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model.fit(xs, ys, epochs=500)

    # Generate predictions for the regression line
    x_vals = np.linspace(min(xs), max(xs), 100)
    y_vals = model.predict(x_vals)
    betas = [b.value()[0] for b in model.trainable_weights]
    print(f"Here is what I learned: {betas[0]}*X + {betas[1]}")
    # Plot the original data points
    plt.scatter(xs, ys, label='Original Data')

    # Plot the regression line
    plt.plot(x_vals, y_vals, color='red', label='Regression Line')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


