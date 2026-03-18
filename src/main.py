import os
import sys
import logging
import argparse
import torch
from datetime import datetime

from config import (
    train_file, test_file, test_file_sample, csv_path, csv_path_sample, batch_size, max_length,
    epoch_num as default_epochs, learning_rate as default_lr,
    save_path as default_save_path,
    pred_path as default_pred_path,
    pred_path_sample as default_pred_path_sample,
    seed as default_seed
)
from train import train
from inference import inference


def setup_logging():
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Query Rewriting with BART")
    parser.add_argument('--epochs', type=int, default=default_epochs,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=default_lr,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=default_seed,
                        help='Random seed')
    parser.add_argument('--model_save_path', type=str, default=default_save_path,
                        help='Path to save the trained model')
    parser.add_argument('--output_file', type=str, default=default_pred_path,
                        help='Path to save inference results')
    parser.add_argument('--output_file_sample', type=str, default=default_pred_path_sample,
                        help='Path to save inference results(sample)')
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help='Skip training and only run inference')
    parser.add_argument('--skip_inference', action='store_true', default=False,
                        help='Skip inference after training')
    parser.add_argument('--skip_eval', action='store_true', default=False,
                        help='Skip evaluation')
    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logging()
    logger.info("Starting Query Rewriting full pipeline")

    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        sys.exit(1)
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        sys.exit(1)

    if not args.skip_train:
        logger.info("===== Training =====")
        logger.info(f"Training parameters: epochs={args.epochs}, lr={args.lr}, "
                    f"batch_size={batch_size}, max_length={max_length}")
        try:
            trained_model = train(
                epochs=args.epochs,
                lr=args.lr,
                save_path=args.model_save_path
            )
            # logger.info(f"Training completed. Model saved to {args.model_save_path}")
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("Skipping training phase")

    if not args.skip_inference:
        logger.info("===== Inference =====")
        logger.info(f"Using model: {args.model_save_path}")
        logger.info(f"Output file: {args.output_file}")
        try:
            inference(
                model_path=args.model_save_path,
                test_file=test_file,
                output_file=args.output_file
            )
            logger.info(f"Inference completed. Results saved to {args.output_file}")
            inference(
                model_path=args.model_save_path,
                test_file=test_file_sample,
                output_file=args.output_file_sample
            )
            logger.info(f"Inference(sample) completed. Results saved to {args.output_file_sample}")
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("Skipping inference phase")

    if not args.skip_eval:
        logger.info("===== Evaluation =====")
        try:
            from eval import eval
            eval(test_file, args.output_file, csv_path)
            eval(test_file_sample, args.output_file_sample, csv_path_sample)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("Skipping evaluation phase")

    logger.info("All steps finished successfully")


if __name__ == "__main__":
    main()