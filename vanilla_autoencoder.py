import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

transform = torchvision.transforms.ToTensor()

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer1 = nn.Linear(
            in_features=784, out_features=128
        )
        self.encoder_layer2 = nn.Linear(
            in_features=128, out_features=128
        )    

    def forward(self, X):
        h1 = self.encoder_layer1(X)
        a1 = F.relu(h1)
        h2 = self.encoder_layer2(a1)
        a2 = F.relu(h2)    

        return a2

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_layer1 = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_layer2 = nn.Linear(
            in_features=128, out_features=784
        )    

    def forward(self, X):
        h1 = self.decoder_layer1(X)
        a1 = F.relu(h1)
        h2 = self.decoder_layer2(a1)
        a2 = F.relu(h2)    

        return a2

class AE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = Encoder()
        self.decode = Decoder()

    def forward(self, X):
        latent_z = self.encode(X)
        x_cap = self.decode(latent_z)

        return x_cap         

train_dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST("./", download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=False)

model = AE()


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 30

for epoch in range(epochs):
    loss = 0
    for batch, labels in train_loader:
        batch = batch.reshape(-1, 784)
        optimizer.zero_grad()

        reconstruction = model(batch)
        train_loss = criterion(batch, reconstruction)
        train_loss.backward()
        optimizer.step()

        loss += train_loss.item()
    loss = loss / len(train_loader)
    print("epoch {}/{}, loss = {:.6f}".format(epoch, epochs, loss))

torch.save(model.state_dict, './ae.pth')

with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 5))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_dataset.data[index].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        test_data = test_dataset.data[index]
        test_data = test_data
        test_data = test_data.float()
        test_data = test_data.view(-1, 784)
        output = model(test_data)
        plt.imshow(output.cpu().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
