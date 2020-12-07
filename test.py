import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import sys


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Layers
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(64*5*5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # define forward propagation here
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    data_path = str(sys.argv[1])
    pth_path = str(sys.argv[2])

    test_transforms = transforms.Compose([transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(data_path, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load(pth_path))
    test(model, device, test_loader)
    
if __name__ == '__main__':
    main()