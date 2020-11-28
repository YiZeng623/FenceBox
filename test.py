from defense import *
cleandata = np.load("./data/clean100data.npy")
cleanlabel = np.load("./data/clean100label.npy")
targets = np.load("./data/random_targets.npy")

test = defend_ET(cleandata[0])