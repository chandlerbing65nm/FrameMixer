# File: datasets/dataset_selection.py

from .affia3k import get_dataloader as affia3k_loader
from .uffia import get_dataloader as uffia_loader
from .mrsffia import get_dataloader as mrsffia_loader

def get_dataloaders(args, transform):
    """
    Selects and returns the training and validation data loaders based on the dataset specified in args.

    Args:
        args: Parsed command-line arguments containing dataset configurations.

    Returns:
        train_dataset: The training dataset.
        train_loader: DataLoader for the training dataset.
        val_dataset: The validation dataset.
        val_loader: DataLoader for the validation dataset.
    """
    if args.dataset == 'affia3k':
        train_dataset, train_loader = affia3k_loader(
            split='train',
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            shuffle=True,
            seed=args.seed,
            class_num=args.num_classes,
            drop_last=True,
            data_path=args.data_path,
            transform=transform
        )
        val_dataset, val_loader = affia3k_loader(
            split='test',
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            shuffle=False,
            seed=args.seed,
            class_num=args.num_classes,
            drop_last=False,
            data_path=args.data_path,
            transform=None  # Typically, no augmentation for validation
        )
    elif args.dataset == 'uffia':
        train_dataset, train_loader = uffia_loader(
            split='train',
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            shuffle=True,
            seed=args.seed,
            class_num=args.num_classes,
            drop_last=True,
            data_path=args.data_path,
            transform=transform
        )
        val_dataset, val_loader = uffia_loader(
            split='test',
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            shuffle=False,
            seed=args.seed,
            class_num=args.num_classes,
            drop_last=False,
            data_path=args.data_path,
            transform=None
        )
    elif args.dataset == 'mrsffia':
        train_dataset, train_loader = mrsffia_loader(
            split='train',
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            shuffle=False,
            seed=args.seed,
            drop_last=False,
            data_path=args.data_path,
            transform=transform
        )
        val_dataset, val_loader = mrsffia_loader(
            split='test',
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            shuffle=False,
            seed=args.seed,
            drop_last=False,
            data_path=args.data_path,
            transform=None
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    return train_dataset, train_loader, val_dataset, val_loader
