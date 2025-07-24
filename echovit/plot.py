import numpy as np
import matplotlib.pyplot as plt

# Kayıpları yükleme
train_losses = np.load("/home/eda/Desktop/EchoViT/output/train_losses.npy")
val_losses = np.load("/home/eda/Desktop/EchoViT/output/val_losses.npy")

iterations = np.linspace(0,5,num=len(train_losses))#DIKKAT DIKKAT 
iterations1 = np.linspace(0,5,num=len(val_losses))#DIKKAT DIKKAT 
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_losses, label="Training Loss")
plt.plot(iterations1, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("training_validation_loss_plot.png")
