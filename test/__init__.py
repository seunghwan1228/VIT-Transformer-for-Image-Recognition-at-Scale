from datahandler.load_data import DataLoader


# Data Loader Tester
data_loader = DataLoader('cifar10', ('train', 'test'), 224, 32, 'standard', 16)
dataset = data_loader.download_data()

train, test = data_loader.split_data(dataset)

print(train, test)

train_ds = data_loader.batch_process_data(train)
test_ds = data_loader.batch_process_data(test)


idx_counter = []
for n, (img, label) in enumerate(train_ds):
    print(img.shape)
    idx_counter.append(n)
print(idx_counter[-1])

