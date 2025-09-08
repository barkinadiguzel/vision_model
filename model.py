# Computer Vision with MNIST Example for GitHub
import torchvision
from torchvision import datasets
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Set device to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- DATASET --------------------
# Get the MNIST train dataset and transform images to tensors
train_data = datasets.MNIST(
    root=".",                 # Save dataset in current directory
    train=True,               # Download the training dataset
    download=True,            # Download if not available
    transform=transforms.ToTensor()  # Convert PIL images to PyTorch tensors (0-1 range)
)

# Get the MNIST test dataset
test_data = datasets.MNIST(
    root=".", 
    train=False,              # Test set
    download=True, 
    transform=transforms.ToTensor()
)

# Check the length of train and test datasets
len(train_data), len(test_data)  # Output: (60000, 10000)

# -------------------- EXPLORING A SAMPLE --------------------
# Get first sample image and label
img = train_data[0][0]   # image tensor [1,28,28]
label = train_data[0][1] # integer label 0-9
print(f"Image:\n {img}") 
print(f"Label:\n {label}")

# Get class names
class_names = train_data.classes  # ['0', '1', ..., '9']

# Visualize first 5 images
for i in range(5):
    img = train_data[i][0]         # image tensor
    img_squeeze = img.squeeze()    # remove channel dimension -> [28,28]
    label = train_data[i][1]
    plt.figure(figsize=(3,3))
    plt.imshow(img_squeeze, cmap="gray")  # plot grayscale image
    plt.title(label)                      # set title as true label
    plt.axis(False)

# -------------------- DATALOADER --------------------
# Wrap datasets in DataLoader for batching
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=32,    # number of samples per batch
    shuffle=True      # shuffle data at each epoch
)
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=32,
    shuffle=False     # don't shuffle test data
)

len(train_dataloader), len(test_dataloader)  # number of batches

# -------------------- MODEL DEFINITION --------------------
class MNIST_model(torch.nn.Module):
    """Convolutional Neural Network for MNIST classification."""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        # First convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,   # number of input channels (1 for MNIST)
                out_channels=hidden_units, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ), 
            nn.ReLU(),                     # activation function
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    # downsample by 2
        )
        # Second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Classifier (fully connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # flatten 2D feature maps to 1D
            nn.Linear(in_features=hidden_units*7*7,  # 28/2 pool -> 14/2 pool -> 7x7
                      out_features=output_shape)     # output logits for 10 classes
        )

    # Forward pass
    def forward(self, x):
        x = self.conv_block_1(x)  # pass through first conv block
        x = self.conv_block_2(x)  # pass through second conv block
        x = self.classifier(x)    # flatten and linear layer
        return x

# -------------------- MODEL INSTANTIATION --------------------
model = MNIST_model(input_shape=1, hidden_units=10, output_shape=10).to(device)

# Try a dummy forward pass to check shapes
dummy_x = torch.rand(size=(1,28,28)).unsqueeze(dim=0).to(device)  # [batch, channel, H, W]
model(dummy_x)

# -------------------- TRAINING LOOP --------------------
# Create loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # multi-class classification loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 5
for epoch in tqdm(range(epochs)):
    train_loss = 0
    model.train()  # set model to training mode
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)  # move data to GPU if available
        y_pred = model(X)                   # forward pass
        loss = loss_fn(y_pred, y)           # compute loss
        train_loss += loss                  # accumulate batch loss
        optimizer.zero_grad()               # reset gradients
        loss.backward()                     # backpropagation
        optimizer.step()                    # update weights
    train_loss /= len(train_dataloader)    # average loss

    # -------------------- TEST LOOP --------------------
    test_loss_total = 0
    model.eval()  # set model to evaluation mode
    with torch.inference_mode():  # disable gradient calculation
        for batch, (X_test, y_test) in enumerate(test_dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)
            test_loss_total += test_loss
        test_loss_total /= len(test_dataloader)
    
    print(f"Epoch: {epoch} | Loss: {train_loss:.3f} | Test loss: {test_loss_total:.3f}")

# -------------------- SINGLE IMAGE PREDICTION --------------------
plt.imshow(test_data[0][0].squeeze(), cmap="gray")  # visualize first test image
model_pred_logits = model(test_data[0][0].unsqueeze(dim=0).to(device))  # add batch dim
model_pred_probs = torch.softmax(model_pred_logits, dim=1)               # convert logits -> probabilities
model_pred_label = torch.argmax(model_pred_probs, dim=1)                 # pick predicted class
model_pred_label

# -------------------- MULTIPLE IMAGE PREDICTION --------------------
num_to_plot = 5
for i in range(num_to_plot):
    img = test_data[i][0]
    label = test_data[i][1]
    model_pred_logits = model(img.unsqueeze(dim=0).to(device))
    model_pred_probs = torch.softmax(model_pred_logits, dim=1)
    model_pred_label = torch.argmax(model_pred_probs, dim=1)
    plt.figure()
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Truth: {label} | Pred: {model_pred_label.cpu().item()}")
    plt.axis(False)

# -------------------- CONFUSION MATRIX --------------------
# Generate predictions for entire test set
model.eval()
y_preds = []
with torch.inference_mode():
    for batch, (X, y) in tqdm(enumerate(test_dataloader)):
        X, y = X.to(device), y.to(device)
        y_pred_logits = model(X)
        y_pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
        y_preds.append(y_pred_labels)
    y_preds = torch.cat(y_preds).cpu()  # concatenate all batches into single tensor

# Compare first 10 predictions to ground truth
test_data.targets[:10], y_preds[:10]

# Setup confusion matrix
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_preds, target=test_data.targets)

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),  # convert tensor to numpy
    class_names=class_names,          # class labels
    figsize=(10,7)                    # figure size
)
plt.show()
