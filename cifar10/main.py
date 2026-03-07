import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

print("Veri seti yükleniyor...")

dataset_path = r"C:\Users\irem\Downloads\archive\cifar10"

train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

labels = sorted(os.listdir(train_path))  # sınıf isimleri

x_train = []
y_train = []

for label_index, label in enumerate(labels):

    folder = os.path.join(train_path, label)

    images = os.listdir(folder)

    for img in images[:1000]:  # hız için sınıf başına 300

        img_path = os.path.join(folder, img)

        image = cv2.imread(img_path)

        image = cv2.resize(image, (32,32))

        x_train.append(image.flatten()/255.0)

        y_train.append(label_index)

x_train = np.array(x_train)
y_train = np.array(y_train)

print("Train veri sayısı:", len(x_train))

x_test = []
y_test = []

for label_index, label in enumerate(labels):

    folder = os.path.join(test_path, label)

    images = os.listdir(folder)

    for img in images[:50]:  # hız için sınıf başına 50

        img_path = os.path.join(folder, img)

        image = cv2.imread(img_path)

        image = cv2.resize(image,(32,32))

        x_test.append(image.flatten()/255.0)

        y_test.append(label_index)

x_test = np.array(x_test)
y_test = np.array(y_test)

print("Test veri sayısı:", len(x_test))

metric = input("Mesafe metriğini seçin (L1 / L2): ")

k = int(input("k değerini girin: "))

test_index = 0
test_vector = x_test[test_index]

distances = []

print("Mesafeler hesaplanıyor...")

for i in range(len(x_train)):

    if metric == "L1":
        distance = np.sum(np.abs(x_train[i] - test_vector))

    elif metric == "L2":
        distance = np.sqrt(np.sum((x_train[i] - test_vector)**2))

    distances.append((distance, y_train[i]))

distances = sorted(distances, key=lambda x: x[0])

neighbors = distances[:k]

neighbor_labels = []

for n in neighbors:
    neighbor_labels.append(n[1])

values, counts = np.unique(neighbor_labels, return_counts=True)

prediction = values[np.argmax(counts)]

print("\nTek görüntü tahmini:", labels[prediction])
print("Gerçek sınıf:", labels[y_test[test_index]])

img = x_test[test_index].reshape(32,32,3)

plt.imshow(img)
plt.title("Tahmin: " + labels[prediction])
plt.axis("off")
plt.show()

print("\nTest seti üzerinde doğruluk hesaplanıyor...")

correct = 0

for t in range(len(x_test)):

    test_vector = x_test[t]

    distances = []

    for i in range(len(x_train)):

        if metric == "L1":
            distance = np.sum(np.abs(x_train[i] - test_vector))

        elif metric == "L2":
            distance = np.sqrt(np.sum((x_train[i] - test_vector)**2))

        distances.append((distance, y_train[i]))

    distances = sorted(distances, key=lambda x: x[0])

    neighbors = distances[:k]

    neighbor_labels = []

    for n in neighbors:
        neighbor_labels.append(n[1])

    values, counts = np.unique(neighbor_labels, return_counts=True)

    prediction = values[np.argmax(counts)]

    if prediction == y_test[t]:
        correct += 1

accuracy = correct / len(x_test)

print("\nDoğruluk oranı (Accuracy):", accuracy)
