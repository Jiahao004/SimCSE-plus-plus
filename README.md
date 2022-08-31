# decoupled_simcse
decoupled_simcse
## environment
cuda10.2
torch=1.12.1
datasets=1.18.3

## run
two scripts

original simcse: run_unsup_examples.sh
decoupled simcse (mean-positives): run_both.sh
decoupled simcse (non-dropout version): run_re.sh

### mean positives
mean the two representation from different dropout pattern for negative contrasting

### non-dropout version
sample one more representation without dropout (full PLM network) for negative contrasting, while using the dropout positives for regulization only
