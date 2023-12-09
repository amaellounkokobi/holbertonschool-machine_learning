#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

lbl = np.array(["apples", "bananas", "oranges", "peaches"])
clr = np.array(['red', 'yellow', "#ff8000", "#ffe5b4"])
per = np.array(["Farrah", "Fred", "Felicia"])
bt = np.zeros(3)

for idx, row in enumerate(fruit):
    plt.bar(per, row, 0.5, label=lbl[idx], color=clr[idx], bottom=bt)
    bt += row

plt.ylim(0,80)
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.legend()
plt.show()
