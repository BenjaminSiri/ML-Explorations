import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0]);
y_train = np.array([300.0, 500.0]);
print(f"x_train = {x_train}");
print(f"y_train = {y_train}");

# m is the number of training examples
print(f"x_train.shape = {x_train.shape}");
m = x_train.shape[0];
print(f"Number of training examples = {m}");
m = len(x_train);
print(f"Number of training examples = {m}");

for i in range(m):
    x_i = x_train[i];
    y_i = y_train[i];
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})");

w = 200;
b = 100;
print(f"w = {w}");
print(f"b = {b}");

def compute_model_output(x, w, b):
    m = x.shape[0];
    f_wb = np.zeros(m);
    for i in range(m):
        f_wb[i] = w * x[i] + b;
    return f_wb;


# Making predictions
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")

tmp_f_wb = compute_model_output(x_train, w, b);

plt.plot(x_train, tmp_f_wb, label='model prediction');
plt.scatter(x_train, y_train, marker='x', c='r', label='actual values');
plt.title("Housing prices");
plt.ylabel("Price in 1000s of dollars");
plt.xlabel("Size in square thousand feet");
plt.show();