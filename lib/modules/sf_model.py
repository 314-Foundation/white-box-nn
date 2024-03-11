def get_model():
    augment = nn.Sequential(
        Noise(mean=0.05, scale=0.25, clip=True, p=0.7),
    )

    latent_dim = 6

    pixel_layer = TwoStepFunction(10)

    conv_layer = ConvLayer(
        sampler=RotationSampler(16),
        n_kernels=2,
        kernel_size=5,
        act=nn.ReLU(),
        # add_bias=True,
        # rescale=True,
    )

    affine_layer = AffineLayer(
        sampler=AffineSampler(32, dim_pose_group=latent_dim),
        inp_shape=IMG_SHAPE,
        feature_shape=(1, 18, 18),
        out_dim=latent_dim,
        act=nn.ReLU(),
        # add_bias=True,
        # rescale=True,
    )

    # head = nn.Linear(
    #     latent_dim,
    #     N_CLASSES,
    #     bias=False,
    #     # bias=True,
    # )
    # with torch.no_grad():
    #     w = head.weight
    #     t0 = torch.tensor([1., 0.1, 0., 0.])
    #     t1 = torch.roll(t0, 2, dims=0)
    #     # t0, t1 = torch.arange(1, -1, step=-0.35), torch.arange(-1, 1, step=0.35)
    #     t = torch.stack([t0, t1], dim=0)
    #     t = t / 5
    #     w.data = t

    # nn.init.eye_(w)
    # w.add_(torch.randn_like(w) * 0.1)

    sparse = TwoPieceLayer(
        sampler=TwoPieceRollSampler(3, mask_grad=True),
        inp_dim=latent_dim,
    )

    backbone = nn.Sequential(
        OrderedDict(
            [
                ("pixel", pixel_layer),
                ("conv", conv_layer),
                ("affine", affine_layer),
                ("sparse", sparse),
            ]
        )
    )

    model = ClsModule(
        learning_rate=3e-3,  # BEST
        # weight_decay=3e-7,# 0.00001,
        weight_decay=3e-6,  # BEST?
        # weight_decay=0.0,
        noise_eps=EPS,
        adv_prob=0.0,
        # adv_prob=1.0,
        val_adv_prob=1.0,
        n_classes=N_CLASSES,
        augment=augment,
        backbone=backbone,
    )
