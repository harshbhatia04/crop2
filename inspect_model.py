from torchvision import models
import torch.nn as nn

model = models.efficientnet_b3()
print("Model features[-1]:")
print(model.features[-1])

print("\nModel features[-2]:")
print(model.features[-2])
