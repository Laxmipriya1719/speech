# test_dataloader.py
from utils.dataset import DataloaderFactory
from types import SimpleNamespace

# Create a mock config for RAVDESS
args = SimpleNamespace(
    database='ravdess',
    meta_csv_file='metadata.csv',  # Make sure this path is correct
    wavdir='wav/archive (2)',      # Make sure this path is correct
    batch_length=96000,
    num_workers=0,
    world_size=1
)

factory = DataloaderFactory(args)
loader = factory.build(state='train', bs=1)

for batch in loader:
    print("Batch loaded:")
    print("Waveform shape:", batch['waveform'][0].shape)
    print("Padding mask shape:", batch['padding_mask'][0].shape)
    print("Emotion:", batch['emotion'][0].item())
    break  # Just load one batch to test
