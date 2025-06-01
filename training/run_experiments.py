

def run_experiment(cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Transforms
    train_t = get_cifar10_transforms('train')
    test_t  = get_cifar10_transforms('test')
    # CIFAR-10 loaders
    train_loader, test_loader = get_cifar10_loaders(
        cfg['batch_size'], cfg['data_dir'], train_t, test_t)
    # Build model
    model = build_sc_resnet(num_classes=cfg['num_classes'],
                            use_adabn=cfg['use_adabn'],
                            use_cbam=cfg['use_cbam'],
                            use_proto=cfg['use_proto']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'],
                          momentum=0.9, weight_decay=5e-4)

    # Training loop
    for epoch in range(cfg['epochs']):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader,
                                     criterion, device)
        print(f"Epoch {epoch+1}/{cfg['epochs']}: "
              f"train_acc={tr_acc:.4f}, val_acc={val_acc:.4f}")

    # CIFAR-10-C evaluation
    print("\n=== CIFAR-10-C Evaluation ===")
    for corr in cfg['corruptions']:
        accs = []
        for sev in range(1, 6):
            cset = CIFAR10C(cfg['c10c_dir'], corruption=corr,
                            severity=sev, transform=test_t)
            cl = DataLoader(cset, batch_size=cfg['batch_size'], shuffle=False)
            _, acc = evaluate(model, cl, criterion, device)
            accs.append(acc)
        print(f"{corr}: avg_acc={sum(accs)/len(accs):.4f}")

    # CIFAR-10-P evaluation
    print("\n=== CIFAR-10-P Evaluation ===")
    for pert in cfg['perturbations']:
        flips = 0
        total = 0
        pset = CIFAR10P(cfg['c10p_dir'], perturbation=pert, transform=test_t)
        loader = DataLoader(pset, batch_size=1, shuffle=False)
        for frames, label in loader:
            # frames: (1, T, C, H, W)
            frames = frames.to(device)  # shape (1,T,C,H,W)
            # evaluate sequence
            preds = []
            with torch.no_grad():
                for t in range(frames.size(1)):
                    out = model(frames[:, t])
                    preds.append(out.argmax(dim=1).item())
            # count flips
            flips += sum(preds[i] != preds[i-1] for i in range(1, len(preds)))
            total += (len(preds) - 1)
        print(f"{pert}: flip_rate={flips/total:.4f}")


if __name__ == '__main__':
    cfg = {
        'data_dir': './data',
        'c10c_dir': './CIFAR-10-C',
        'c10p_dir': './CIFAR-10-P',
        'batch_size': 128,
        'num_classes': 10,
        'lr': 0.1,
        'epochs': 20,
        'use_adabn': True,
        'use_cbam': False,
        'use_proto': False,
        'corruptions': [
            'gaussian_noise','motion_blur','brightness',
            'contrast','fog','snow','jpeg_compression'
        ],
        'perturbations': [
            'brightness','translate','rotate','scale','shear'
        ],
    }
    run_experiment(cfg)

