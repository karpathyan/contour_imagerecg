import matplotlib.pyplot as plt
from numpy import random

for i in range(50):
    Z = random.random((50, 50))  # Test data
    plt.imshow(Z, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Turn off the axis
    fname = "rand" + str(i) + ".png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)  # Save the figure without padding
    # plt.show()  # Uncomment this if you want to display the image as well
    plt.close()  # Close the figure to free up memory
