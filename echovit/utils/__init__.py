
import os
import typing

import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
import tqdm


def loadvideo(filename:str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError
    
    # Open video
    capture         = cv2.VideoCapture(filename)

    frame_count     = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width     = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height    = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v               = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)# (F ,H, W, C) 


    # Read video frame by frame  
    for count in range(frame_count):
        ret, frame  = capture.read()# If ret is True, reading is succesful 
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame           = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB 
        v[count, :, :]  = frame

    v = v.transpose((3, 0, 1, 2)) # (C, F, H, W)   
    return v

def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    """Computes mean and std from samples from a Pytorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.

    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """
    

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    for x, _ in tqdm.tqdm(dataloader):# x:video 
        x = torch.as_tensor(x)
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


def clip_collate(batch):
    """
    Collate function that converts a batch of samples—
    each of which may contain one or more clips—
    into a single tensor of shape (total_clips, C, T, H, W).

    Args:
        batch: list of tuples (video, target)
               - video: np.ndarray of shape (C, T, H, W) or (N, C, T, H, W)
               - target: label associated with the original video

    Returns:
        videos: torch.Tensor of shape (sum_N, C, T, H, W)
        targets: list of targets, one per clip
    """
    videos = []
    targets = []
    filenames = []

    for video, [filename, target] in batch:
        # If video has 4 dimensions, it’s a single clip (C, T, H, W)
        if video.ndim == 4:
            videos.append(torch.from_numpy(video))
            targets.append(target)
            filenames.append(filename)
        else:
            # video.ndim == 5 → multiple clips (N, C, T, H, W)
            # append each clip separately
            for clip in video:
                videos.append(torch.from_numpy(clip))
                targets.append(target)
                filenames.append(filename)

    # Stack all clips into one batch dimension
    videos = torch.stack(videos, dim=0)  # shape = (total_clips, C, T, H, W)
    targets = torch.tensor(targets, dtype=torch.float32)
    return videos, (filenames, targets) 