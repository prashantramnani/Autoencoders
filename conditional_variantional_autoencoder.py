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
            in_features=794, out_features=400
        )

        self.mean = nn.Linear(
            in_features=400, out_features=20
        )
        
        self.variance = nn.Linear(
            in_features = 400, out_features=20
        )

    def forward(self, X, y):
        
        x1 = torch.cat((X, y), axis=1)

        h1 = self.encode_layer1(x1)
        a1 = F.relu(h1)

        return a1

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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode_layer1 = nn.Linear(
            in_features=30, out_features=400
        )

        self.decode_layer2 = nn.Linear(
            in_features=400, out_features=784
        )

    def forward(self, X, y):
        x1 = torch.cat((X, y), axis=1)
        h1 = self.decode_layer1(x1)
        a1 = F.relu(h1)
        h2 = self.decode_layer2(a1)
        a2 = F.sigmoid(h2)

        return a2    

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = Encoder()
        self.latent = latent()
        self.decode = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std    

    def forward(self, X, y):
        a = self.encode(X, y)    
        mu, var = self.latent(a)
        z = self.reparameterize(mu, var)
        reconstruction = self.decode(z, y)

        return reconstruction, mu, var


def idx2onehot(idx, n=10):

    assert idx.shape[1] == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)

    return onehot


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
        for i, (data, labels) in enumerate(test_loader):
            
            recon_batch, mu, logvar = model(data.reshape(-1, 784), idx2onehot(labels.view(-1, 1)))
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(128, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results_cvae/reconstruction_' + str(epoch) + '.png', nrow=n)    
   

train_dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST("./", download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=True)

model = CVAE()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 20

if 'train' in argumentList:
    
    for epoch in range(epochs):
        loss = 0
        Mse = 0
        KL = 0
        bce = 0
        for batch, labels in train_loader:
            batch = batch.reshape(-1, 784)
            optimizer.zero_grad()

            y = idx2onehot(labels.view(-1, 1))

            reconstruction, mu, logvar = model(batch, y)

            KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #torch.mean(0.5*torch.sum( torch.exp(model.var)+ 0*torch.log(model.var**2 + 1e-8)  + model.mean**2 + model.var**2 - 1, axis=1), axis=0)
            bce_loss = F.binary_cross_entropy(reconstruction, batch, reduction='sum')

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

    torch.save(model.state_dict(), './cvae.pth')
else:
    model.load_state_dict(torch.load('./cvae.pth'))




z = torch.randn(1, 20)

y = torch.randint(0, 10, (1, 1)).to(dtype=torch.long)
print(f'Generating a {y.item()}')

y = idx2onehot(y)
# z = torch.cat((z, y), dim=1)

reconstructed_img = model.decode(z, y)
img = reconstructed_img.view(28, 28).data

plt.figure()
plt.imshow(img, cmap='gray')
plt.show()