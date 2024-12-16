"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
import pandas as pd
from argparse import ArgumentParser
from evaluation_metrics import ssim, nmse, psnr
import h5py

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.data.transforms import center_crop
from fastmri.pl_modules import FastMriDataModule, VarNetModule


import os
os.chdir("/content/drive/MyDrive/exercise_3-1/")

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations,
    )
    # mask = create_mask_for_mask_type(
    #     12, args.center_fractions, args.accelerations,
    # )
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform(mask_func=mask)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        #test_path=args.test_path,
        test_path = "/content/drive/MyDrive/Colab_Notebooks/fastMRIdataOnecoil/multicoil_test/",
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # print how large the training/validation set is
    test_set = data_module.test_dataloader().dataset
    print("Size of trainingset:", len(test_set))
    #print("Size of trainingset:", test_set[3])
    #testloader = load_data()

    # ------------
    # model
    # ------------

    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module, 
        ckpt_path='/content/drive/MyDrive/exercise_3-1/varnet/varnet_demo/checkpoints/attention_epoch=19-step=18820-v2.ckpt')
    return


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("save_model/fastmri_dirs.yaml")
    num_gpus = 1
    batch_size = 1

    # set defaults based on optional directory config
    # data_path = "/projects/0/gpuuva035/reconstruction"
    data_path = "/content/drive/MyDrive/Colab_Notebooks/fastMRIdataOnecoil"
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"


    # client arguments
    parser.add_argument(
        "--mode",
        default="test",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[2],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        # mask_type="equispaced_fraction",  # VarNet uses equispaced mask
        mask_type="random",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,
    )

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=4,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.001,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        # strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TESTING
    # ---------------------
    cli_main(args)


def evaluate_test_data_quantitatively(datapath, reconpath):
    #print(datapath)
    #print(reconpath)
    for root, _, files in os.walk(datapath):
        scores_ssim = []
        scores_nmse = []
        scores_psnr = []
        for file in files:
            print(file)
            gt_data = h5py.File(datapath + file)
            recon_data = h5py.File(reconpath + file)
            # calculate average SSIM of scan
            num_slices = len(recon_data['reconstruction'])
            print(num_slices)

           # for i in range(num_slices):
            rec = recon_data['reconstruction']
            gt = np.abs(gt_data['reconstruction_rss'])
            gt = center_crop(gt,rec.shape[-2:])
            #print("shape of reconstructions:",rec.shape)
            #print("shape of ground truth:",gt.shape)

            score_nmse = nmse(gt / np.max(gt), rec / np.max(rec))

            score_ssim = ssim(gt / np.max(gt), rec / np.max(rec))

            score_psnr = psnr(gt / np.max(gt), rec / np.max(rec))

            scores_ssim.append(score_ssim)
            scores_nmse.append(score_nmse)
            scores_psnr.append(score_psnr)

    # print scores
    print("Average SSIM:", np.mean(scores_ssim))
    print("STD SSIM", np.std(scores_ssim))
    print("Median SSIM", np.median(scores_ssim))

    print("Average NMSE:", np.mean(scores_nmse))
    print("STD NMSE", np.std(scores_nmse))
    print("Median NMSE", np.median(scores_nmse))

    print("Average PSNR:", np.mean(scores_psnr))
    print("STD PSNR", np.std(scores_psnr))
    print("Median PSNR", np.median(scores_psnr))
    # show boxplot
    #plt.boxplot(scores_ssim)
    #plt.title("SSIM")
    #plt.show()
    #plt.savefig('SSIM-comparison.png')

    data = {
    'SSIM': scores_ssim,
    'NMSE': scores_nmse,
    'PSNR': scores_psnr
    }

    df = pd.DataFrame(data)
    df.to_csv('Sparse.csv', index=False)

    return


def evaluate_test_data_qualitatively(datapath, reconpath):
    for root, _, files in os.walk(datapath):
        scores = []
        for file in files:
            # check random slice
            gt = h5py.File(datapath + file)['reconstruction_rss'][20]
            recon = h5py.File(reconpath + file)['reconstruction'][20]

            gt = center_crop(gt,recon.shape[-2:])
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # check ground truth image
            ax[0,0].imshow(np.abs(gt), cmap='gray')
            print(gt.shape)
            #plt.colorbar()
            #plt.show()

            # check reconstruction
            ax[0,1].imshow(recon, cmap='gray')
            print(recon.shape)
            #plt.colorbar()
            #plt.show()
            for ax in ax.flat:
              ax.axis('off')
            plt.tight_layout()

            name = "gt_recon-comparison" + str(file) + ".png"
            plt.savefig(name)
    return


if __name__ == "__main__":
    # run testing the network
    run_cli()
    #datapath = '/projects/0/gpuuva035/reconstruction/multicoil_test/'
    datapath = "/content/drive/MyDrive/Colab_Notebooks/fastMRIdataOnecoil/multicoil_test/"
    reconpath = 'varnet/varnet_demo/reconstructions/'
    # quantitativaly evaluate data
    evaluate_test_data_quantitatively(datapath, reconpath)
    # qualitatively
    #evaluate_test_data_qualitatively(datapath, reconpath)
