1. Update D
2. Update RS
3. Update G

```
  train:
    json_file: /home/admin/workspace/tuan/s2igan_model/dataset/train_flower_en2vi.json
    img_path: /home/admin/workspace/tuan/s2igan_model/dataset/image_oxford/image_oxford
    audio_path: /home/admin/workspace/tuan/s2igan_model/dataset/oxford_audio/oxford
    input_size: ${data.general.input_size}
    n_fft: ${data.general.n_fft}
    n_mels: ${data.general.n_mels}
    win_length: ${data.general.win_length}
    hop_length: ${data.general.hop_length}
  test:
    json_file: /home/admin/workspace/tuan/s2igan_model/dataset/test_flower_en2vi.json
    img_path: /home/admin/workspace/tuan/s2igan_model/dataset/image_oxford/image_oxford
    audio_path: /home/admin/workspace/tuan/s2igan_model/dataset/oxford_audio/oxford
    input_size: ${data.general.input_size}
    n_fft: ${data.general.n_fft}
    n_mels: ${data.general.n_mels}
    win_length: ${data.general.win_length}
    hop_length: ${data.general.hop_length}
```
