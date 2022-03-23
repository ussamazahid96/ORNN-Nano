import os
import sys
import logging


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


class TrainingEpochMeters(object):
    def __init__(self):
        self.loss = AverageMeter()
        self.accuracy = AverageMeter()


class EvalEpochMeters(object):
    def __init__(self):
        self.loss = AverageMeter()
        self.accuracy = AverageMeter()

class Logger(object):

    def __init__(self, output_dir_path, dry_run):
        self.output_dir_path = output_dir_path
        self.log = logging.getLogger('log')
        self.log.setLevel(logging.INFO)

        # Stout logging
        out_hdlr = logging.StreamHandler(sys.stdout)
        out_hdlr.setFormatter(logging.Formatter('%(message)s'))
        out_hdlr.setLevel(logging.INFO)
        self.log.addHandler(out_hdlr)

        # Txt logging
        if not dry_run:
            file_hdlr = logging.FileHandler(os.path.join(self.output_dir_path, 'log.txt'))
            file_hdlr.setFormatter(logging.Formatter('%(message)s'))
            file_hdlr.setLevel(logging.INFO)
            self.log.addHandler(file_hdlr)
            self.log.propagate = False

    def info(self, arg):
        self.log.info(arg)

    def training_batch_cli_log(self, epoch_meters, epoch, tot_ep, batch, tot_batches):
        self.info('Epoch: [{0}/{1}][{2}/{3}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                         .format(epoch, tot_ep, batch, tot_batches,
                                 loss=epoch_meters.loss,
                                 acc=epoch_meters.accuracy))


    def eval_batch_cli_log(self, epoch_meters, batch, tot_batches):
        self.info('Batch: [{0}/{1}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                         .format(batch, tot_batches,
                                 loss=epoch_meters.loss,
                                 acc=epoch_meters.accuracy))