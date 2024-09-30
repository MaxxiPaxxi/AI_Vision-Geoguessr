import matplotlib.pyplot as plt

# Example dictionary
data = {
    2: 9.541,
    3: 9.486,
    4: 9.469,
    5: 9.467,
    6:9.482,
    7:9.5,
    8:9.488,
    9:9.5,
    10:9.52,
    11:9.514,
    12:9.495,
    13:9.501,
    14:9.507,
    15:9.51,
}

data = {
    2: 0.92,
    3: 0.81,
    4: 0.67,
    5: 0.71,
    6:0.62,
    7:0.5,
    8:0.51,
    9:0.47,
    10:0.4,
    11:0.405,
    12:0.42,
    13:0.398,
    14:0.389,
    15:0.4,
}

x_updated = list(data.keys())
y_updated = list(data.values())

# Plotting with linear line
plt.figure(figsize=(10, 6))
plt.plot(x_updated, y_updated, '-o', color='red')
plt.xlabel('values of k')
plt.ylabel('Accuracy')
plt.title('Accuracy with varying k-neighbours')
plt.grid(True)
plt.show()