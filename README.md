# Environment Diversification with Multi-head Neural Network for Invariant Learning

This code implements EDNIL from the following paper accepted by NeurIPS 2022.
> Environment Diversification with Multi-head Neural Network for Invariant Learning

## Environment

To create the environment with Python 3.8.12:

```
torch==1.7.1
torchvision==0.8.2
numpy==1.21.2
pandas==1.4.1
```

## Training and Evaluation

The implementations lie in `./src`. To train and evaluate EDNIL, run `main.py` with arguments of configured hyper-parameters. As an alternative, a single json file can also be used to set up values more concisely. For instance, run the following commands to obtain the results of CMNIST:

```bash=
cd src/
python main.py --config_file ../config/cmnist.json
# or python main.py --dataset cmnist --val_ratio 0.1 --joint_iters 5 ...
```

For more instructions to tune hyper-parameters, please refer to Appendix B.2 in the paper.

## Citation

If you find our work useful to your research, please consider citing our paper using the following bibtex entry:

```
@inproceedings{huang2022environment,
  title={Environment Diversification with Multi-head Neural Network for Invariant Learning},
  author={Bo-Wei Huang and Keng-Te Liao and Chang-Sheng Kao and Shou-De Lin},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022}
}
```
