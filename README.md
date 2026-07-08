# BTRFusion: Misaligned Medical Image Fusion via Bidirectional Translation and Registration

This is the official pytorch implementation of the paper 'BTRFusion: Misaligned Medical Image Fusion via Bidirectional Translation and Registration'.

By Yuan Chang and Zheng Li

---

## Environment

```bash
Python >= 3.9
PyTorch >= 2.0
CUDA >= 11.8
```

---

## Dataset

Experiments are conducted on two anatomical-functional medical image datasets:

| Dataset   | Modality                |
| --------- | ----------------------- |
| PET-MRI   | Structural + Functional |
| SPECT-MRI | Structural + Functional |

Dataset source:

http://www.med.harvard.edu/AANLIB/home.html

---

## Testing

```bash
python test.py
```

Results will be saved to:

```text
result/
├── /
```

---

## Contact

For questions and collaborations, feel free to open an issue or contact us via email.
