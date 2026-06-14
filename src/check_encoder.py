import pickle

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print("Classes:")
print(encoder.classes_)