import torch
import os
import torchvision
from torch.autograd import Variable

G = torch.nn.Sequential(
    torch.nn.Linear(128, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 28*28),
    torch.nn.Sigmoid()
)

D = torch.nn.Sequential(
    torch.nn.Linear(28*28, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 1),
    torch.nn.Sigmoid()
)

######################### Main Function.........
dataset = torchvision.datasets.MNIST('./MNIST', train=True, transform=torchvision.transforms.ToTensor(), target_transform=None, download=True)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)

# Optimizers
g_optimizer = torch.optim.Adam(G.parameters(), 0.0002)
d_optimizer = torch.optim.Adam(D.parameters(), 0.0002)
# a,x  = np.array( i, images for i, images in data_loader)
"""Train generator and discriminator."""
fixed_noise = Variable(torch.randn(10, 128))  # For Testing
print(fixed_noise.shape)
for epoch in range(200):
    for i, images in enumerate(data_loader):
        # ===================== Train D =====================#
        # print(images[602])
        # print(images[0].shape)
        images = Variable(images[0])
        # print(images)
        # print(images.shape)
        images = images.view(100, 1 * 28 * 28)

        noise = Variable(torch.randn(images.size(0), 128))
        fake_images = G(noise)
        d_loss = -torch.mean(torch.log(D(images) + 1e-8) + torch.log(1 - D(fake_images) + 1e-8))

        # Optimization
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===================== Train G =====================#
        noise = Variable(torch.randn(images.size(0), 128))
        fake_images = G(noise)
        g_loss = -torch.mean(torch.log(D(fake_images) + 1e-8))

        # Optimization
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print and Save
        if (i + 1) % 10 == 0:
            # print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f'% (epoch + 1, 200, i + 1, len(data_loader), d_loss.data[0], g_loss.data[0]))
            fake_images = G(fixed_noise)
            fake_images = fake_images.view(10, 1, 28, 28)
            torchvision.utils.save_image(fake_images.data, 'C:/Users/asthe/OneDrive - KNOU/study/testint/50line_gan_images/generatedimage-%d-%d.png' % (epoch + 1, i + 1))