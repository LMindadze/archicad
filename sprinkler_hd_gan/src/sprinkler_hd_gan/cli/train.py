from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sprinkler_hd_gan.data.dataset import SprinklerLayoutDataset
from sprinkler_hd_gan.models.networks import GlobalGenerator, MultiScaleDiscriminator
from sprinkler_hd_gan.util import load_yaml_config


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.Conv2d) and m.bias is not None:
        nn.init.constant_(m.bias, 0)


def _downsample(x: torch.Tensor) -> torch.Tensor:
    return nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)


def _at_scale(t: torch.Tensor, scale_idx: int) -> torch.Tensor:
    cur = t
    for _ in range(scale_idx):
        cur = _downsample(cur)
    return cur


def main() -> None:
    p = argparse.ArgumentParser(description="Train pix2pixHD-style sprinkler layout GAN.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--data", type=Path, required=True, help="Dataset root with train/ and val/ splits.")
    p.add_argument("--out", type=Path, default=Path("runs/exp01"))
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    cfg = load_yaml_config(args.config)
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = int(args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        tqdm.write(f"Device: CUDA {torch.cuda.get_device_name(0)} ({props.total_memory // (1024**3)} GiB)")
    else:
        tqdm.write(
            "Device: CPU (slow). Install CUDA PyTorch: "
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

    haz_cfg = cfg["hazard_encoding"]
    hazard_mode = str(haz_cfg["mode"])
    in_ch = 3 + (1 if hazard_mode == "scalar" else 3)

    tcfg = cfg["training"]
    g = GlobalGenerator(
        in_ch,
        3,
        ngf=int(tcfg["ngf"]),
        n_downsampling=int(tcfg["n_downsample_global"]),
        n_blocks=int(tcfg["n_blocks_global"]),
    ).to(device)
    g.apply(_init_weights)

    d_in = in_ch + 3
    d = MultiScaleDiscriminator(
        d_in,
        ndf=int(tcfg["ndf"]),
        n_layers=3,
        num_D=int(tcfg["num_d_scales"]),
    ).to(device)
    d.apply(_init_weights)

    opt_g = torch.optim.Adam(g.parameters(), lr=float(tcfg["lr_g"]), betas=(float(tcfg["beta1"]), float(tcfg["beta2"])))
    opt_d = torch.optim.Adam(d.parameters(), lr=float(tcfg["lr_d"]), betas=(float(tcfg["beta1"]), float(tcfg["beta2"])))

    criterion = nn.MSELoss()
    l1 = nn.L1Loss()

    aug = cfg["augmentation"] if cfg.get("augmentation") else {}
    train_ds = SprinklerLayoutDataset(
        args.data,
        "train",
        hazard_mode=hazard_mode,
        paper_background_tint=bool(haz_cfg.get("paper_background_tint_for_film", False)),
        augment={k: bool(v) for k, v in aug.items()},
    )
    val_ds = SprinklerLayoutDataset(
        args.data,
        "val",
        hazard_mode=hazard_mode,
        paper_background_tint=bool(haz_cfg.get("paper_background_tint_for_film", False)),
        augment={},
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    args.out.mkdir(parents=True, exist_ok=True)

    num_epochs = int(tcfg["num_epochs"])
    lambda_gan = float(tcfg["lambda_gan"])
    lambda_feat = float(tcfg["lambda_feat"])
    lambda_l1 = float(tcfg["lambda_l1"])

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        g.train()
        d.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs}")
        for batch in pbar:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            y = y * 2.0 - 1.0

            # --- D ---
            with torch.no_grad():
                fake0 = g(x)
            fake0 = fake0.detach()

            opt_d.zero_grad(set_to_none=True)
            loss_d = 0.0
            for si in range(d.num_D):
                h_x = _at_scale(x, si)
                h_y = _at_scale(y, si)
                h_f = _at_scale(fake0, si)
                pred_real = d.nets[si](torch.cat([h_x, h_y], dim=1))
                pred_fake = d.nets[si](torch.cat([h_x, h_f], dim=1))
                loss_d += criterion(pred_real[0], torch.ones_like(pred_real[0]))
                loss_d += criterion(pred_fake[0], torch.zeros_like(pred_fake[0]))
            loss_d = loss_d / d.num_D
            loss_d.backward()
            opt_d.step()

            # --- G ---
            opt_g.zero_grad(set_to_none=True)
            fake = g(x)
            loss_g_adv = 0.0
            fm = 0.0
            g_scales = []
            r_scales = []
            for si in range(d.num_D):
                h_x = _at_scale(x, si)
                h_y = _at_scale(y, si)
                h_fk = _at_scale(fake, si)
                pr = d.nets[si](torch.cat([h_x, h_y], dim=1))
                pf = d.nets[si](torch.cat([h_x, h_fk], dim=1))
                g_scales.append(pf)
                r_scales.append(pr)
                loss_g_adv += criterion(pf[0], torch.ones_like(pf[0]))

            for pf, pr in zip(g_scales, r_scales):
                for a, b in zip(pf[1], pr[1]):
                    fm += l1(a, b)

            loss_g_adv = lambda_gan * (loss_g_adv / d.num_D)
            nfeat = sum(len(s[1]) for s in g_scales)
            fm = lambda_feat * (fm / max(1, nfeat))
            l1_loss = lambda_l1 * l1(fake, y)
            total_g = loss_g_adv + fm + l1_loss
            total_g.backward()
            opt_g.step()

            global_step += 1
            if global_step % 20 == 0:
                pbar.set_postfix(
                    loss_d=float(loss_d.detach().cpu()),
                    loss_g=float(total_g.detach().cpu()),
                )

        ckpt_path = args.out / f"epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "generator": g.state_dict(),
                "discriminator": d.state_dict(),
                "cfg": cfg,
                "in_ch": in_ch,
                "hazard_mode": hazard_mode,
            },
            ckpt_path,
        )

        # quick val L1
        g.eval()
        with torch.no_grad():
            acc = 0.0
            n = 0
            for vb in val_loader:
                vx = vb["input"].to(device)
                vy = vb["target"].to(device) * 2.0 - 1.0
                pred = g(vx)
                acc += float(l1(pred, vy).cpu())
                n += 1
        tqdm.write(f"val_l1_epoch_{epoch}={acc/max(1,n):.5f} saved={ckpt_path}")


if __name__ == "__main__":
    main()
