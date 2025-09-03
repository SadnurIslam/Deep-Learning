from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    model = build_model()
    model.compile(optimizer='adam', loss='mse')
    
    (trainX, trainY), (valX, valY), (testX, testY) = prepare_train_val_test()
    
    history = model.fit(trainX, trainY,
                        validation_data=(valX, valY),
                        epochs=50,
                        verbose=1)
    
    test_pred = model.predict(testX)
    test_mse = mean_squared_error(testY, test_pred)
    print("Test MSE:", test_mse)
    
    plot_original_vs_predicted(model)

def prepare_train_val_test():
    x, y = data_process()
    total_n = len(x)
    
    indices = np.arange(total_n)
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]
    
    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)
    
    trainX = x[:train_n]
    trainY = y[:train_n]
    valX = x[train_n:train_n + val_n]
    valY = y[train_n:train_n + val_n]
    testX = x[train_n + val_n:]
    testY = y[train_n + val_n:]
    
    return (trainX, trainY), (valX, valY), (testX, testY)

def data_process():
    n = 10000
    x = np.random.uniform(-10, 10, n)
    y = my_polynomial(x)
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    return x, y

def my_polynomial(x):
    return 5 * x**2 + 10*x - 2

def build_model():
    inputs = Input((1,))
    h1 = Dense(16, activation='relu')(inputs)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)
    outputs = Dense(1)(h3)

    model = Model(inputs, outputs)
    model.summary()
    return model

def plot_original_vs_predicted(model):
    x_plot = np.linspace(-10, 10, 500).reshape(-1, 1)
    y_true = my_polynomial(x_plot)
    y_pred = model.predict(x_plot)
    
    plt.figure(figsize=(8,6))
    plt.plot(x_plot, y_true, label='Original f(x)', linewidth=2)
    plt.plot(x_plot, y_pred, label='Predicted f(x)', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Original vs Predicted Function')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
\end{lstlisting}

\end{document}