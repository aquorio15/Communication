import os
import random
import numpy as np
import time
import argparse
import torch
import logging
import configs
import torch.nn as nn
from tqdm import tqdm
from dataset import *
from model_CNN import *
from model_transformer import *
from scheduler import *
import torch.functional as F
from datetime import timedelta
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import re
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)

logger = logging.getLogger(__name__)

eval_l, train_l = [], []


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, "module") else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # num_classes = 10 if args.dataset == "cifar10" else 100
    num_classes = 11
    config = CONFIGS[args.model_type]
    # model = CNN(
    #     in_channels=2, out_channels=64, out_channels_new=32, num_classes=num_classes
    # )
    model = VisionTransformer(config, zero_head=True, num_classes=num_classes)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(
        test_loader,
        desc="Validating... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
    )
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        x = x.float()
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy, eval_losses.avg


def train(args, model):
    """Train the model"""
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    Train_Dataset = Communicationdataset(
        root="/nfsshare/Amartya/EMNLP-WACV/communication_journal",
    )
    train_size = int(0.7 * len(Train_Dataset))
    valid_size = len(Train_Dataset) - train_size
    trainset, testset = torch.utils.data.random_split(
        Train_Dataset, [train_size, valid_size]
    )
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=8,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )
    else:
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(
            train_loader,
            desc="Training (X / X Steps) (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            x = x.float()
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)"
                    % (global_step, t_total, losses.val)
                )
                writer.add_scalar(
                    "train/loss", scalar_value=losses.val, global_step=global_step
                )
                writer.add_scalar(
                    "train/lr",
                    scalar_value=scheduler.get_lr()[0],
                    global_step=global_step,
                )
                if global_step % args.eval_every == 0:
                    accuracy, eval_los = valid(
                        args, model, writer, test_loader, global_step
                    )
                    train_l.append(losses.avg)
                    eval_l.append(eval_los)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break
    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--name", required=True, help="Name of this run. Used for monitoring."
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="The output directory where checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        choices=[
            "ViT-B_16",
            "ViT-B_32",
            "ViT-L_16",
            "ViT-L_32",
            "ViT-H_14",
            "R50-ViT-B_16",
        ],
        default="ViT-B_16",
        help="Which variant to use.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=224,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=224, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--eval_every",
        default=500,
        type=int,
        help="Run prediction on validation set every so many steps."
        "Will always run one evaluation at the end of training.",
    )

    parser.add_argument(
        "--learning_rate",
        default=3e-2,
        type=float,
        help="The initial learning rate for SGD.",
    )
    parser.add_argument(
        "--weight_decay", default=0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--num_steps",
        default=50000,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--decay_type",
        choices=["cosine", "linear"],
        default="cosine",
        help="How to decay the learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=300,
        type=int,
        help="Step of training to perform learning rate warmup for.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s, n_gpu: %s" % (args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)
    np.save(f"{args.output_dir}/train.npy", np.array(train_l))
    np.save(f"{args.output_dir}/val.npy", np.array(eval_l))


if __name__ == "__main__":
    main()
