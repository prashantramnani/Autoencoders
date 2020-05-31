import sys

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
from torchvision.utils import save_image
import matplotlib.pyplot as plt

transform = torchvision.transforms.ToTensor()

argumentList = sys.argv

# torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode_layer1 = nn.Linear(
            in_features=784, out_features=400
        )

        self.mean = nn.Linear(
            in_features=400, out_features=20
        )
        
        self.variance = nn.Linear(
            in_features = 400, out_features=20
        )

    def forward(self, X):
        h1 = self.encode_layer1(X)
        a1 = F.relu(h1)

        return a1#self.mean(a1), self.variance(a1)

class latent(nn.Module):
    def __init__(self):
        super().__init__()

        self.mean = nn.Linear(
            in_features=400, out_features=20
        )
        
        self.variance = nn.Linear(
            in_features = 400, out_features=20
        )

    def forward(self, X):

        m = self.mean(X)
        v = self.variance(X)
        
        return m, v

# class Sample(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.m = normal.Normal(0, 1)

#     def forward(self, mean, variance):
        
#         std = torch.exp(0.5*variance)
#         z = mean + std*self.m.sample((20, ))

#         return z

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode_layer1 = nn.Linear(
            in_features=20, out_features=400
        )

        self.decode_layer2 = nn.Linear(
            in_features=400, out_features=784
        )

    def forward(self, X):
        h1 = self.decode_layer1(X)
        a1 = F.relu(h1)
        h2 = self.decode_layer2(a1)
        a2 = F.sigmoid(h2)

        return a2    

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = Encoder()
        self.latent = latent()
        # self.sample = Sample()
        self.decode = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)

        m = normal.Normal(0, 1)
        x = m.sample((mu.shape[0], 20))
        # print("m",x)
        eps = torch.randn_like(std)
        # print("eps", eps)
        return mu + x*std    

    def forward(self, X):

        a = self.encode(X)    
        mu, var = self.latent(a)
        z = self.reparameterize(mu, var)
        reconstruction = self.decode(z)

        return reconstruction, mu, var

# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)
#         self.fc3 = nn.Linear(20, 400)
#         self.fc4 = nn.Linear(400, 784)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


train_dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST("./", download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=True)

model = VAE()
print("MODEL", model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 20

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            
            recon_batch, mu, logvar = model(data.reshape(-1, 784))
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(128, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)



if 'train' in argumentList:
    
    for epoch in range(epochs):
        loss = 0
        Mse = 0
        KL = 0
        bce = 0
        for batch, labels in train_loader:
            batch = batch.reshape(-1, 784)
            optimizer.zero_grad()

            reconstruction, mu, logvar = model(batch)
            # print("SHAPE", torch.sum( torch.log(1/model.var)  + model.mean**2 + model.var**2 - 1, axis=1).shape)
            KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #torch.mean(0.5*torch.sum( torch.exp(model.var)+ 0*torch.log(model.var**2 + 1e-8)  + model.mean**2 + model.var**2 - 1, axis=1), axis=0)
            # mse_loss = criterion(batch, reconstruction)
            bce_loss = F.binary_cross_entropy(reconstruction, batch, reduction='sum')

            # train_loss = loss_function(reconstruction, batch, mu, logvar)
            # train_loss.backward()
            train_loss = bce_loss + KL_loss


            train_loss.backward()

            loss += train_loss.item()
            # Mse += mse_loss.item()
            KL += KL_loss.item()
            bce += bce_loss.item()

            optimizer.step()
        loss = loss / len(train_loader)
        # Mse = Mse/ len(train_loader)
        KL = KL / len(train_loader)
        bce = bce / len(train_loader)

        test(epoch)
        

        print("epoch {}/{}, loss = {:.6f}, kl_loss = {:.6f}, bce_loss = {:.6f}".format(epoch+1, epochs, loss, KL, bce))

    torch.save(model.state_dict(), './vae.pth')
else:
    model.load_state_dict(torch.load('./trained_models/vae.pth'))



# with torch.no_grad():
#     number = 10
#     plt.figure(figsize=(20, 5))
#     for index in range(number):
#         # display original
#         ax = plt.subplot(2, number, index + 1)
#         plt.imshow(test_dataset.data[index + 10].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#         # display reconstruction
#         ax = plt.subplot(2, number, index + 1 + number)
#         test_data = test_dataset.data[index + 10]
#         test_data = test_data.float()
#         test_data = test_data.view(-1, 784)
#         # print("type", type(test_data))
#         output, mu, var = model(test_data)
#         # print("shape", type(output))
#         plt.imshow(output.reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()



