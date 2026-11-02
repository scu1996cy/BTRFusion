# BTRFusion: Misaligned Medical Image Fusion via Bidirectional Translation and Registration

This is the official pytorch implementation of the paper 'BTRFusion: Misaligned Medical Image Fusion via Bidirectional Translation and Registration,' published in the IEEE International Conference on Acoustics, Speech and Signal Processing. [Paper](https://ieeexplore.ieee.org/abstract/document/11460786)

---

## Environment

```bash
Python >= 3.9
PyTorch >= 2.0
CUDA >= 11.8
```

---

## Dataset

Experiments are conducted on two anatomical-functional (PET-MRI and SPECT-MRI) medical image datasets from [Harvard Medical Dataset](http://www.med.harvard.edu/AANLIB/home.html).

---

## Testing

Before running the commands, please ensure that the dataset has been correctly placed.

```bash
python test.py
```

Results will be saved to:

```text
result/
├── /
```

---

## Correction

There is a minor numerical error in the reported FLOPs and parameter count in our paper. We contacted IEEE to request a correction or an erratum. IEEE replied that conference papers are published as received and that no post-publication edits or errata are made. Therefore, we provide the corrected values here. This correction affects only FLOPs and parameters, not any textual content.

| Method | FLOPs(G) (T+R+F) | FLOPs(G) (R+F) | Params (T+R+F) | Params (R+F) |
|---|---:|---:|---:|---:|
| ReCoNet | [-] | [9.79] | [-] | [0.20] |
| UMF | [103.34] | [47.3] | [12.31] | [0.94] |
| IMF | [75.03] | [18.99] | [13.44] | [2.07] |
| PAMRFuse | [95.01] | [38.97] | [13.91] | [2.54] |
| BSAFusion | [11.86] | [8.95] | [9.69] | [6.18] |
| Ours | [69.33] | [13.29] | [19.93] | [8.56] |

---

## Contact

If you have any questions, feel free to contact changyuan@stu.scu.edu.cn


---

## Citation

If you find this work useful, please cite:

```bibtex
@INPROCEEDINGS{11460786,
  author    = {Chang, Yuan and Li, Zheng},
  booktitle = {ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title     = {BTRFusion: Misaligned Medical Image Fusion via Bidirectional Translation and Registration},
  year      = {2026},
  pages     = {6301--6305},
  doi       = {10.1109/ICASSP55912.2026.11460786}
}
```

---

## Reference and Acknowledgments

[VoxelMorph](https://github.com/voxelmorph/voxelmorph)
[TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
