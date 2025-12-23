"""Dataset utilities for IID and non-IID data partitioning - FIXED."""

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar10_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return transform_train, transform_test


def load_cifar10_datasets(data_path="./data"):
    transform_train, transform_test = get_cifar10_transforms()
    
    trainset = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )
    
    return trainset, testset


def create_iid_partitions(dataset, num_clients):
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    samples_per_client = num_samples // num_clients
    client_indices = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else num_samples
        client_indices.append(indices[start_idx:end_idx].tolist())
    
    return client_indices

def create_dirichlet_partitions(dataset, num_clients, alpha=0.5, num_classes=10):
    """Chia dữ liệu theo phân phối Dirichlet (LDA)."""
    labels = np.array(dataset.targets)
    client_idcs = [[] for _ in range(num_clients)]
    
    # Với mỗi class, chia các sample cho các client theo tỷ lệ Dirichlet
    for c in range(num_classes):
        idx_k = np.where(labels == c)[0]
        np.random.shuffle(idx_k)
        
        # Sinh tỷ lệ phân chia (proportions)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Cân bằng lại để tránh client không có dữ liệu (có thể tinh chỉnh thêm)
        proportions = np.array([p * (len(idx_j) < len(labels) / num_clients) for p, idx_j in zip(proportions, client_idcs)])
        proportions = proportions / proportions.sum()
        
        # Tính điểm cắt index
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Chia và gán
        client_splits = np.split(idx_k, proportions)
        for client_idx, split in enumerate(client_splits):
            client_idcs[client_idx].extend(split)

    return client_idcs

def create_non_iid_partitions(dataset, num_clients, num_classes=10, classes_per_client=2):
    class_indices = {i: [] for i in range(num_classes)}
    
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    for class_idx in class_indices:
        np.random.shuffle(class_indices[class_idx])
    
    client_indices = [[] for _ in range(num_clients)]
    client_classes = []
    
    for i in range(num_clients):
        selected_classes = np.random.choice(num_classes, classes_per_client, replace=False)
        client_classes.append(selected_classes)
    
    for class_idx in range(num_classes):
        clients_with_this_class = [i for i, classes in enumerate(client_classes) 
                                   if class_idx in classes]
        
        if len(clients_with_this_class) == 0:
            continue
        
        samples = class_indices[class_idx]
        samples_per_client = len(samples) // len(clients_with_this_class)
        
        for i, client_id in enumerate(clients_with_this_class):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < len(clients_with_this_class) - 1 else len(samples)
            client_indices[client_id].extend(samples[start_idx:end_idx])
    
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    return client_indices


def get_client_dataloader(dataset, client_indices, batch_size=32, is_train=True,
                          num_workers=0, pin_memory=True):
    subset = Subset(dataset, client_indices)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=is_train,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )


def prepare_datasets(client_id, num_clients, batch_size=32, partition_type="iid", alpha=0.5,
                    num_workers=0, pin_memory=True):
    """
    Prepare datasets for a client - FIXED to handle dynamic client_id.
    
    Args:
        client_id: Client ID (can be from 0 to num_supernodes-1)
        num_clients: Total number of logical clients (for partitioning)
        batch_size: Batch size
        partition_type: "iid" or "non-iid"
        num_workers: Number of workers
        pin_memory: Pin memory
    
    Returns:
        trainloader, testloader
    """
    trainset, testset = load_cifar10_datasets()
    
    # Create partitions based on num_clients
    if partition_type == "iid":
        train_partitions = create_iid_partitions(trainset, num_clients)
        test_partitions = create_iid_partitions(testset, num_clients)
    elif partition_type == "dirichlet": 
        # Lưu ý: Thường chỉ cần chia Train set là Non-IID, Test set có thể giữ IID hoặc chia theo
        train_partitions = create_dirichlet_partitions(trainset, num_clients, alpha=alpha)
        # Test set có thể chia tương tự hoặc dùng IID tùy thiết kế của bạn
        test_partitions = create_iid_partitions(testset, num_clients)
    else:
        train_partitions = create_non_iid_partitions(trainset, num_clients, classes_per_client=2)
        test_partitions = create_non_iid_partitions(testset, num_clients, classes_per_client=2)
    
    # FIX: Map client_id to partition index using modulo
    # This handles cases where num_supernodes > num_clients
    partition_idx = client_id % num_clients
    
    trainloader = get_client_dataloader(
        trainset, train_partitions[partition_idx], 
        batch_size=batch_size, is_train=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    testloader = get_client_dataloader(
        testset, test_partitions[partition_idx], 
        batch_size=batch_size, is_train=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return trainloader, testloader