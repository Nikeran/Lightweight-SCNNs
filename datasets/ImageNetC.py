class ImageNetC(Dataset):
    """
    A PyTorch Dataset for ImageNet‐C.  
    We expect IMAGENET_C_ROOT to contain directories like `gaussian_blur/1/…`, `gaussian_blur/2/…`, etc.
    We walk all corruption types and severities, collect all image paths, then map filename → label using the dictionary above.
    """
    def __init__(self, root: str, filename_to_label: dict, transform=None):
        """
        root: path to “imagenet-c” folder (e.g. "/…/imagenet/imagenet-c")
        filename_to_label: dict mapping "ILSVRC2012_val_00000001.JPEG" -> integer class (0–999)
        transform: torchvision transforms to apply to each corrupted image
        """
        super().__init__()
        self.root = Path(root)
        self.filename_to_label = filename_to_label
        self.transform = transform

        self.samples = []  # will hold tuples: (image_path, label, corruption_name, severity_int)

        # Walk through every “corruption_name” directory under root:
        for corruption_dir in sorted(self.root.iterdir()):
            if not corruption_dir.is_dir():
                continue
            corruption_name = corruption_dir.name  # e.g. "gaussian_blur"
            # Inside each corruption directory, there should be five subfolders: "1", "2", … "5"
            for severity_folder in sorted(corruption_dir.iterdir()):
                if not severity_folder.is_dir():
                    continue
                try:
                    severity = int(severity_folder.name)  # 1 through 5
                except ValueError:
                    # skip if folder isn’t named “1”–“5”
                    continue

                # Now gather all JPEGs in this severity folder:
                for img_path in severity_folder.glob("*.JPEG"):
                    fname = img_path.name  # e.g. "ILSVRC2012_val_00000001.JPEG"
                    if fname not in self.filename_to_label:
                        # If a filename isn’t found in the clean‐val mapping, skip it
                        continue
                    label = self.filename_to_label[fname]
                    self.samples.append((str(img_path), label, corruption_name, severity))

        print(f"✔️  Collected {len(self.samples)} ImageNet‐C samples across all corruptions/severities.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, corruption_name, severity = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # We return a tuple of (tensor_img, label, corruption_name, severity)
        return img, label, corruption_name, severity