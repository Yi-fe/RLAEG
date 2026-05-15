# MalConv2 Components

This directory contains PyTorch implementations related to the MalConv2 model family, including low-memory convolution helpers and several MalConv variants.

## Files

```text
AvastStyleConv.py      Avast-style CNN variant
LowMemConv.py          Fixed-memory convolution base class
MalConv.py             Original MalConv-style model
MalConvGCT_nocat.py    Global Channel Gating variant
MalConvML.py           Multi-layer MalConv variant
binaryLoader.py        Binary loading helpers
checkpoint.py          Gradient-checkpointing helper
```

These files are model components used by RLAEG's classifier side. They are not the main training entry points for the reinforcement-learning attack workflow.

## Citation

If you use the MalConv GCT implementation, cite the original work:

```bibtex
@inproceedings{malconvGCT,
  author = {Raff, Edward and Fleshman, William and Zak, Richard and Anderson, Hyrum and Filar, Bobby and Mclean, Mark},
  booktitle = {The Thirty-Fifth AAAI Conference on Artificial Intelligence},
  title = {{Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection}},
  year = {2021},
  url = {https://arxiv.org/abs/2012.09390},
}
```
