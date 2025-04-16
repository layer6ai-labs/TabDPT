# TabDPT: Scaling Tabular Foundation Models

## Installation
```
git clone git@github.com:layer6ai-labs/TabDPT.git
cd TabDPT
pip install -e .
```

## Example Usage 
Please take a look at `tests/cls_example.py` and `tests/reg_example.py`
For better performance, please increase `context_size` or increase `n_ensembles` to trade off speed and accuracy

## Updates

### Update April 2025: New Model
**Version 1.1 is now available.** We have improved the prediction performance of TabDPT through increased training stability.

### Update December 2024: Faster Inference
Added support for flash attention (with bf16 precision) and compile flag. Both are enabled to True by default and should lead to a significant speed-up.

## Citation
```
@article{ma2024tabdpt,
  title={TabDPT: Scaling Tabular Foundation Models},
  author={Ma, Junwei and Thomas, Valentin and Hosseinzadeh, Rasa and Kamkari, Hamidreza and Labach, Alex and Cresswell, Jesse C and Golestan, Keyvan and Yu, Guangwei and Volkovs, Maksims and Caterini, Anthony L},
  journal={arXiv preprint arXiv:2410.18164},
  year={2024}
}
```

## Roadmap
- [ ] Release other model sizes
- [ ] Release training code
