from torchvision import transforms

def get_transforms(split: str = 'train') -> transforms.Compose:
    """
    Returns CIFAR-10 transforms for 'train' or 'test' split.
    """
    if split == 'train':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
        ])
    elif split == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
def get_imagenet_transforms(split: str = 'train') -> transforms.Compose:
    """
    Returns ImageNet transforms for 'train' or 'test' split.
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    if split == 'train':
        return transforms.Compose([
            # random crop and resize to 224×224
            transforms.RandomResizedCrop(224),
            # random horizontal flip
            transforms.RandomHorizontalFlip(),
            # small random rotation
            transforms.RandomRotation(15),
            # color jitter
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            # convert to tensor
            transforms.ToTensor(),
            # normalize to ImageNet stats
            transforms.Normalize(imagenet_mean, imagenet_std),
            # random erasing for regularization
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value='random'
            ),
        ])
    elif split == 'test':
        return transforms.Compose([
            # resize shorter side to 256
            transforms.Resize(256),
            # center crop to 224×224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    
def get_tiny_imagenet_transforms(split: str = 'train') -> transforms.Compose:
    """
    Returns Tiny ImageNet transforms for 'train' or 'test' split.
    """
    tiny_imagenet_mean = (0.4802, 0.4481, 0.3975)
    tiny_imagenet_std  = (0.2770, 0.2691, 0.2821)

    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(tiny_imagenet_mean, tiny_imagenet_std),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value='random'
            ),
        ])
    elif split == 'test':
        return transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(tiny_imagenet_mean, tiny_imagenet_std),
        ])
