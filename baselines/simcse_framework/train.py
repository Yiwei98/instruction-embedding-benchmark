import torch
from tqdm import tqdm


class EmbeddingTrainer():
    def __init__(self, data_loader, model, optimizer, save_path, epochs=1):
        super().__init__()
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_path = save_path

    def train_epoch(self):
        loss_list = []
        data_loader = tqdm(self.data_loader, desc="Iteration")
        for step, inputs in enumerate(data_loader):
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()

            loss = self.model(**inputs)

            assert not torch.isnan(loss).any().item()
            loss_list.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

            if (step + 1) % 12500 == 0:
                print(f"Step [{step+1}], Loss: {sum(loss_list) / len(loss_list)}")
                break

        return sum(loss_list) / len(loss_list)

    def train(self):
        for epoch in range(self.epochs):
            loss = self.train_epoch()
            if epoch < self.epochs - 1:
                self.model.save_checkpoint(f'{self.save_path}_epoch{epoch+1}')
            else:
                self.model.save_checkpoint(f'{self.save_path}')

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss:.4f}")