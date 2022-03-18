import _pickle as pickle

a_file = open("uda/ds2.pkl", "rb")
unpickler = pickle.Unpickler(a_file)
# if file is not empty scores will be equal
# to the value unpickled
scores = unpickler.load()
print("Done")
#output = pickle.load(a_file)