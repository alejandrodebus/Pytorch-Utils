import torch

# Pseudo-example of how to use k_folds

for train_idx, test_idx in k_folds(n_splits = num_folds):
    dataset_train = NNDataset(indices = train_idx)
    dataset_test = NNDataset(indices = test_idx)
    train_loader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = batch_size_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset = dataset_test, batch_size = batch_size_test, **kwargs)
    for epoch in range(1, num_epochs + 1):
        train(model, optimizer, epoch, device, train_loader, log_interval)
        test(model, device, test_loader)
