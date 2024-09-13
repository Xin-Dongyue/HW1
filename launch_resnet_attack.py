import argparse
import os
from resnet_attack_todo import ResnetPGDAttacker
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch
from torch import optim

parser = argparse.ArgumentParser(description="Adversarial Training on Resnet50 model")
parser.add_argument('--eps', type=float, help='maximum perturbation for PGD attack', default=8 / 255)
parser.add_argument('--alpha', type=float, help='step size for PGD attack', default=2 / 255)
parser.add_argument('--steps', type=int, help='number of steps for PGD attack', default=20)
parser.add_argument('--batch_size', type=int, help='batch size for PGD attack', default=100)
parser.add_argument('--batch_num', type=int, help='number of batches on which to run PGD attack', default=None)
parser.add_argument('--results', type=str, help='name of the file to save the results to', required=True)
parser.add_argument('--resultsdir', type=str, help='name of the folder to save the results to', default='results')
parser.add_argument('--seed', type=int, help='set manual seed value for reproducibility, default 1234',
                    default=1234)
parser.add_argument('--epochs', type=int, help='number of epochs for training', default=10)
args = parser.parse_args()

RESULTS_DIR = args.resultsdir
RESULTS_PATH = os.path.join(RESULTS_DIR, args.results)
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if args.seed:
    SEED = args.seed
    torch.manual_seed(SEED)
else:
    SEED = torch.seed()

EPS = args.eps
ALPHA = args.alpha
STEPS = args.steps
BATCH_SIZE = args.batch_size
BATCH_NUM = args.batch_num if args.batch_num else 1281167 // BATCH_SIZE + 1

print('Loading model...')
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
preprocess = weights.transforms()

attacker = ResnetPGDAttacker(model, DataLoader, steps=STEPS, eps=EPS, alpha=ALPHA)

print('Loading data...')
ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)


def preprocess_img(example):
    example['image'] = preprocess(example['image'])
    return example


ds = ds.filter(lambda example: example['image'].mode == 'RGB')
ds = ds.map(preprocess_img)
ds = ds.shuffle(seed=SEED)
ds = ds.take(BATCH_NUM * BATCH_SIZE)
dset_loader = DataLoader(ds, batch_size=BATCH_SIZE)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def adversarial_train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_adv_loss = 0.0
        for i, inputs in enumerate(dataloader):
            images, labels = inputs['image'].to(attacker.device), inputs['label'].to(attacker.device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            adv_images = attacker.pgd_attack(images, labels)
            adv_outputs = model(adv_images)
            adv_loss = criterion(adv_outputs, labels)

            total_loss = (loss + adv_loss) / 2
            total_loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_adv_loss += adv_loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Adv Loss: {total_adv_loss:.4f}')

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'adv_trained_model.pth'))

adversarial_train(model, dset_loader, optimizer, args.epochs)