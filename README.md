# decoupled_simcse
![Fig1. Decoupled simcse balance the loss for both subnetworks during contrastive learning](https://github.com/Jiahao004/decoupled_simcse/blob/main/1.png)

decoupled_simcse
## environment
cuda10.2
datasets==1.18.3


bert-base-uncased: torch==1.10.2; bert-large-uncased: torch==1.12.1

## run
two scripts

original simcse: run_unsup_examples.sh
decoupled simcse (non-dropout version): run_re.sh

### non-dropout version
sample one more representation without dropout (full PLM network) for negative contrasting, while using the dropout positives for regulization only
