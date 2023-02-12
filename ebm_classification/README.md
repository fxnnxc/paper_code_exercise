

# Energy-based Model with Classifier 

## Train Classifier with Pretrained EBM Models 

```bash 
cd ebm_classification
pip install -e .

export data=mnist

export loss=default
bash shell classifier.sh

```


## Define Another Loss Type

See [emb_pkg/loss.py](emb_pkg/loss.py).

You can run the experiment with different loss easily 

```bash 
export data=mnist
export loss=energy
bash shell classifier.sh
```
