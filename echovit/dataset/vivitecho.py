"""Custom CEchoNet-Dynamic Dataset."""

import os
import collections
import pandas as pd

import numpy as np
import torchvision
import echovit

from echovit.utils import loadvideo

class ViViTecho(torchvision.datasets.VisionDataset):
    def __init__(self, root=None, split="train", 
                 mean=0., std=1.,
                 length=32, period=2,
                 clips=1,
                 external_data_dir=None, external_video_stats=None,
                 oversample=True):
        

        if root is None:
            root = echovit.config.DATA_DIR

        super().__init__(root)
        self.split = split.upper()

        self.mean           = mean
        self.std            = std
        self.length         = length
        self.period         = period
        self.clips          = clips
        self.external_dir   = external_data_dir
        self.external_video_stats = external_video_stats
        self.oversample     = oversample

         # Initialize attributes
        self.fnames     = []
        self.outcome    = []
        self.header     = []

        self.frames     = collections.defaultdict(list)  # Stores frame numbers

        # Load dataset components
        self.load_video_labels()
        # Load and filter traces for labeled data only
        if self.split not in ("EXTERNAL_TEST","EXT"):
            self.load_trace_indices()  
            self.filter_videos_with_traces()

    
    def __getitem__(self,index):

        if self.split in ("EXTERNAL_TEST","EXT"):
            video = os.path.join(self.external_dir, self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = loadvideo(video).astype(np.float32)

        # Apply normalization
        if self.external_video_stats and self.fnames[index] in self.external_video_stats:
            mean, std = self.external_video_stats[self.fnames[index]]
            mean = np.array(mean).reshape(3, 1, 1, 1)
            std = np.array(std).reshape(3, 1, 1, 1)
            video = (video - mean) / std
        else: 

            if isinstance(self.mean, (float, int)):
            # If mean is a single value, subtract it from all pixels 
                video -= self.mean
            else:
                video -= self.mean.reshape(3, 1, 1, 1)

            if isinstance(self.std, (float, int)):
                # If std is a single value, divide all pixels by it
                video /= self.std
            else:
                video /= self.std.reshape(3, 1, 1, 1)   

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if f < length * self.period:
            missing = length * self.period - f
            reps    = (missing // f) + 1          # how many extra loops we need
            extra   = np.concatenate([video] * reps, axis=1)[:, :missing, :, :]
            video   = np.concatenate((video, extra), axis=1)
            c, f, h, w = video.shape

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets (Extract relevant information for training/testing)
        target = []

        # Retrieve the filename of the current video
        file_name = self.fnames[index]
        EF        = float(self.outcome[index][self.header.index("EF")])
        target.append(self.fnames[index])  # Append the filename to the target list
        target.append(EF/100)   #  Append EF value

        # Select clips from video based on starting indices
        video_all = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)

    
        is_minority = self.oversample and ((EF < 40) or (EF > 70)) and self.split=="TRAIN"

        clips_per_video = 20 if is_minority else 1

        # If only one clip is selected, return it as a single tensor 
        if clips_per_video == 1:
            video = video_all[0]
        elif clips_per_video == 20:
            max_start = f - (length - 1) * self.period
            clips_per_video = 20
            replace = max_start < clips_per_video        # yetersiz pencere varsa tekrarlı seç
            start_positions = np.random.choice(max_start,
                                    clips_per_video,
                                    replace=replace)

            video_clips =  tuple(video[:, s + self.period * np.arange(length), :, :] for s in start_positions)
            video = np.stack(video_clips)
        elif self.clips == "all" and clips_per_video == 1:
            # Stack multiple clips into a single tensor with shape (N,C,F,H,W) 
            video = np.stack(video_all)
        else:
            print("error in clips")
            
            
    
        return video, target
    
    def __len__(self):
        return len(self.fnames)

    def load_video_labels(self):
        
        """Load video file names and labels.

        - train/val/test  ➜ read from FileList.csv
        - external_test   ➜ use every *.avi file found in the specified folder
        """
        if self.split in ("EXTERNAL_TEST", "EXT"):  # external test mode
            ext_dir = (
                self.external_data_dir
                if self.external_data_dir is not None
                else os.path.join(self.root, "ExternalTest")  # default fallback
            )
            if not os.path.isdir(ext_dir):
                raise FileNotFoundError(f"External-test folder not found: {ext_dir}")

            # Accept both .avi and .mp4, case-insensitive
            self.fnames = sorted([
                f for f in os.listdir(ext_dir)
                if f.lower().endswith((".avi", ".mp4"))
            ])

            # Dummy labels
            self.header = ["FileName", "EF", "EDV", "ESV"]
            self.outcome = [[np.nan, np.nan, np.nan, np.nan] for _ in self.fnames]
            self.external_dir = ext_dir
            return
        
        file_list_path  = os.path.join(self.root, "FileList.csv")
        data            = pd.read_csv(file_list_path)
        
        # Normalize the 'Split' column to uppercase
        data["Split"]   = data["Split"].str.upper()

        # Filter by dataset split (train/val/test/all)
        if self.split != "ALL":
            data        = data[data["Split"] == self.split]

        # Store column headers and filenames
        self.header     = data.columns.tolist()
        self.fnames     = [
            fn + ".avi" if os.path.splitext(fn)[1] == "" else fn
            for fn in data["FileName"].tolist()
        ]
        self.outcome    = data.values.tolist()

    def load_trace_indices(self):
        """
        Load only the frame indices (Large / Small) from `VolumeTracings.csv`.

        We deliberately ignore the polygon coordinates (X1,Y1,X2,Y2) to:
        1. Reduce memory usage, coordinates can consume hundreds of MB if
        loaded for all videos.
        2. Speed up dataset initialization when segmentation masks are *not*
        required for the current task (e.g., pure EF regression).

        If later you need the full contours, create a separate loader or
        extend this function to cache coordinates on-demand.
        """
        trace_file = os.path.join(self.root, "VolumeTracings.csv")
        with open(trace_file) as f:
            header = f.readline().strip().split(",")
            # Expected header format
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:
                # Unpack filename and frame number; skip coordinate columns
                filename, *_coords, frame = line.strip().split(',')
                frame = int(frame)

                # Append frame index if not already present
                if frame not in self.frames[filename]:
                    self.frames[filename].append(frame)

        # Ensure chronological order: first index = systolic (Small), last = diastolic (Large)
        for fn in self.frames:
            self.frames[fn].sort(key=int)
    def filter_videos_with_traces(self, min_frames: int = 2):
        """
        Keep only videos that have at least `min_frames` traced frames.

        Parameters
        ----------
        min_frames : int
            Minimum number of traced frames required (default = 2).
            Use `min_frames=1` to simply discard videos with *no* traces,
            while retaining single‑frame cases if EF‑only regression is fine.
        """
        valid_mask = [len(self.frames[f]) >= min_frames for f in self.fnames]
        self.fnames   = [f for f, ok in zip(self.fnames, valid_mask) if ok]
        self.outcome  = [o for o, ok in zip(self.outcome, valid_mask) if ok]
