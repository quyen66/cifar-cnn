import warnings

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


def create_iid_partitions(dataset, num_clients, seed=42):
    # seed truyền vào để reproducible
    rng = np.random.default_rng(seed)
    num_samples = len(dataset)
    indices = rng.permutation(num_samples)
    samples_per_client = num_samples // num_clients
    client_indices = []

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (
            start_idx + samples_per_client
            if i < num_clients - 1
            else num_samples
        )
        client_indices.append(indices[start_idx:end_idx].tolist())

    return client_indices


def create_dirichlet_partitions(
    dataset,
    num_clients,
    alpha=0.5,
    num_classes=10,
    seed=42,               #  thêm seed
    min_partition_size=10, # thêm min_partition_size
):
    """Chia dữ liệu theo phân phối Dirichlet (LDA) — chuẩn Flower DirichletPartitioner.

    Tham chiếu:
        Yurochkin et al., "Bayesian Nonparametric Federated Learning of Neural Networks"
        https://arxiv.org/abs/1905.12022

    Args:
        dataset:            torchvision Dataset có thuộc tính .targets
        num_clients:        số client
        alpha:              tham số Dirichlet — nhỏ = non-IID mạnh hơn (0.1 cực non-IID, 1.0 vừa)
        num_classes:        số class (CIFAR-10 = 10)
        seed:               seed cố định để reproducible — BẮT BUỘC cho luận văn
        min_partition_size: số mẫu tối thiểu mỗi client phải có; nếu không đạt sẽ retry
    """
    #  Dùng Generator riêng thay vì global numpy state
    rng = np.random.default_rng(seed)

    #  Đọc labels qua .targets thay vì enumerate(dataset) — nhanh hơn rất nhiều
    labels = np.array(dataset.targets)
    n_total = len(labels)

    # Retry loop — đảm bảo không có partition nào < min_partition_size
    max_retries = 10
    for attempt in range(max_retries):
        client_idcs = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idx_k = np.where(labels == c)[0]
            rng_attempt = np.random.default_rng(seed + attempt * 100 + c)
            rng_attempt.shuffle(idx_k)

            # Sample proportions từ Dirichlet
            proportions = rng_attempt.dirichlet(np.repeat(alpha, num_clients))

            # Self-balancing: zero out client đã đủ quota (n_total / num_clients)
            # → giống Flower DirichletPartitioner(self_balancing=True)
            quota = n_total / num_clients
            proportions = np.array([
                p * (len(idx_j) < quota)
                for p, idx_j in zip(proportions, client_idcs)
            ])

            # Tránh chia cho 0 nếu tất cả client đều đã đầy
            psum = proportions.sum()
            if psum == 0:
                proportions = np.ones(num_clients) / num_clients
            else:
                proportions = proportions / psum

            # Tính điểm cắt và chia
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_splits = np.split(idx_k, split_points)

            for client_idx, split in enumerate(client_splits):
                client_idcs[client_idx].extend(split.tolist())

        # Kiểm tra min_partition_size
        sizes = [len(c) for c in client_idcs]
        if min(sizes) >= min_partition_size:
            break  # Đạt yêu cầu, thoát retry loop

        if attempt < max_retries - 1:
            warnings.warn(
                f"[Dirichlet] Attempt {attempt+1}: min partition size = {min(sizes)} "
                f"< {min_partition_size}. Retrying with seed offset...",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"[Dirichlet] Sau {max_retries} lần thử, min partition size = {min(sizes)} "
                f"vẫn < {min_partition_size}. Cân nhắc tăng alpha hoặc giảm num_clients.",
                RuntimeWarning,
                stacklevel=2,
            )

    return client_idcs


def create_non_iid_partitions(
    dataset,
    num_clients,
    num_classes=10,
    classes_per_client=2,
    seed=42,  
):
    """Chia dữ liệu theo Pathological Non-IID (label quantity skew).

    Mỗi client nhận dữ liệu từ đúng `classes_per_client` class được chọn ngẫu nhiên.
    Samples của mỗi class được chia đều cho tất cả client chọn class đó.

    Tham chiếu:
        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data" (FedAvg paper), AISTATS 2017.
        → Đây là chiến lược "sort and partition" / "pathological non-IID" gốc.

    Args:
        dataset:            torchvision Dataset có thuộc tính .targets
        num_clients:        số client
        num_classes:        số class (CIFAR-10 = 10)
        classes_per_client: số class mỗi client được nhận (mặc định 2)
        seed:               seed cố định để reproducible — BẮT BUỘC cho luận văn
    """
    # Generator riêng, không ảnh hưởng global numpy state
    rng = np.random.default_rng(seed)

    # KHÔNG dùng enumerate(dataset) — cực chậm vì trigger transform pipeline!
    # Thay bằng đọc trực tiếp .targets (list of int, không cần load ảnh)
    labels = np.array(dataset.targets)

    class_indices = {}
    for c in range(num_classes):
        idx = np.where(labels == c)[0].copy()
        rng.shuffle(idx)
        class_indices[c] = idx.tolist()

    # Mỗi client chọn ngẫu nhiên `classes_per_client` class
    client_classes = []
    for _ in range(num_clients):
        selected = rng.choice(num_classes, classes_per_client, replace=False)
        client_classes.append(selected.tolist())

    # Kiểm tra class nào không được chọn bởi client nào → cảnh báo
    for c in range(num_classes):
        clients_with_class = [i for i, cls in enumerate(client_classes) if c in cls]
        if len(clients_with_class) == 0:
            warnings.warn(
                f"[NonIID] Class {c} không được chọn bởi bất kỳ client nào! "
                f"Dữ liệu class này sẽ bị bỏ qua. Cân nhắc tăng classes_per_client "
                f"hoặc đổi seed.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Phân phối samples của mỗi class cho các client chọn nó
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        clients_with_class = [i for i, cls in enumerate(client_classes) if c in cls]
        if not clients_with_class:
            continue

        samples = class_indices[c]
        n = len(samples)
        samples_per = n // len(clients_with_class)

        for i, client_id in enumerate(clients_with_class):
            start = i * samples_per
            end = start + samples_per if i < len(clients_with_class) - 1 else n
            client_indices[client_id].extend(samples[start:end])

    # Shuffle trong mỗi client để tránh ordering bias khi train
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def get_client_dataloader(
    dataset,
    client_indices,
    batch_size=32,
    is_train=True,
    num_workers=0,
    pin_memory=True,
):
    subset = Subset(dataset, client_indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def prepare_datasets(
    client_id,
    num_clients,
    batch_size=32,
    partition_type="iid",
    alpha=0.5,
    num_workers=0,
    pin_memory=True,
    seed=42,  # seed truyền thống nhất xuống tất cả hàm
):
    """Chuẩn bị dataloader cho một client trong vòng FL.

    Args:
        client_id:      ID của client hiện tại (do Flower server cấp)
        num_clients:    Tổng số client logic dùng để tạo partition
        batch_size:     Batch size cho training
        partition_type: "iid" | "non-iid" | "dirichlet"
        alpha:          Tham số Dirichlet (chỉ dùng khi partition_type="dirichlet")
        num_workers:    Số worker cho DataLoader
        pin_memory:     Pin memory cho DataLoader
        seed:           Seed cố định cho toàn bộ quá trình partition — BẮT BUỘC

    Returns:
        trainloader, testloader

    LƯU Ý cho luận văn:
        partition_idx = client_id % num_clients
        → Thiết kế này giả định num_supernodes == num_clients (ví dụ cả hai = 50).
        Nếu num_supernodes > num_clients, nhiều client vật lý sẽ dùng chung partition,
        đây là simulation artifact — không phản ánh FL thực tế. Cần khai báo rõ
        trong phần Experimental Setup của luận văn.
    """
    trainset, testset = load_cifar10_datasets()

    if partition_type == "iid":
        train_partitions = create_iid_partitions(trainset, num_clients, seed=seed)
        test_partitions = create_iid_partitions(testset, num_clients, seed=seed + 1)

    elif partition_type == "dirichlet":
        # Train set: Non-IID theo Dirichlet
        # Test set: IID — cho phép đánh giá global model công bằng (chuẩn FL research)
        train_partitions = create_dirichlet_partitions(
            trainset, num_clients, alpha=alpha, seed=seed
        )
        test_partitions = create_iid_partitions(testset, num_clients, seed=seed + 1)

    else:  # "non-iid"
        # Pathological non-IID: 2 classes/client, chuẩn FedAvg paper (McMahan 2017)
        train_partitions = create_non_iid_partitions(
            trainset, num_clients, classes_per_client=2, seed=seed
        )
        test_partitions = create_non_iid_partitions(
            testset, num_clients, classes_per_client=2, seed=seed + 1
        )

    # Map client_id → partition index
    # Assumption: num_supernodes == num_clients → mỗi client nhận đúng 1 partition riêng.
    # Nếu num_supernodes != num_clients, cần khai báo rõ trong luận văn.
    partition_idx = client_id % num_clients

    trainloader = get_client_dataloader(
        trainset,
        train_partitions[partition_idx],
        batch_size=batch_size,
        is_train=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = get_client_dataloader(
        testset,
        test_partitions[partition_idx],
        batch_size=batch_size,
        is_train=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader