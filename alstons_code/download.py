import pathlib
import numpy as np
from moabb import datasets
from moabb.paradigms import FixedIntervalWindowsProcessing

DURATION = 1
SAMPLE_RATE = 10

def download():
    base_save_dir = pathlib.Path(__file__).parent / "processed"
    base_save_dir.mkdir(exist_ok=True)

    processing = FixedIntervalWindowsProcessing(
        fmin=5, fmax=95,
        length=DURATION,
        stride=DURATION,
        resample=SAMPLE_RATE,
    )

    registry = [
        # datasets.AlexMI,
        # datasets.BNCI2014_001,
        # datasets.BNCI2014_002,
        # datasets.BNCI2014_004,
        # datasets.BNCI2015_001,
        # datasets.BNCI2015_004,
        datasets.Cho2017,
        datasets.GrosseWentrup2009,
        datasets.Lee2019_MI,
        datasets.Ofner2017,
        datasets.PhysionetMI,
        datasets.Schirrmeister2017,
        datasets.Shin2017A,
        datasets.Shin2017B,
        datasets.Weibo2014,
        # datasets.Zhou2016,
    ]

    for Dataset in registry:
        dataset_name = Dataset.__name__.lower()
        dataset_dir = base_save_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        windows, labels, metadata = processing.get_data(dataset=Dataset())
        windows = (windows - np.mean(windows)) / np.std(windows)
        windows = windows.astype(np.float32)
        assert np.isfinite(windows).all()

        ticks = int(DURATION * SAMPLE_RATE)
        assert windows.shape[-1] >= ticks
        windows = windows[..., :ticks]

        # Limit to 100 trials
        windows = windows[:100, :, :]
        labels = labels[:100]

        for i in range(windows.shape[0]):
            trial_data = windows[i, :, :]
            trial_dir = dataset_dir / f"trial_{i+1}"  
            trial_dir.mkdir(exist_ok=True)  
            trial_name = f"{dataset_name}_trial_{i+1}"
            np.save(trial_dir / f"{trial_name}.npy", trial_data)  
            print(f"Processed {dataset_name}/trial_{i+1}/{trial_name}: {trial_data.shape} ({trial_data.dtype})")

if __name__ == "__main__":
    download()