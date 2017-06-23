`STFTDataset` uses a JSON manifest to know which audio files to load and their
respective targets. The file should have the following format:

```
{"key": "path/to/audio1.wav", "target": [1, 14]}
{"key": "path/to/audio2.wav", "target": [3, 11]}
```

