# nlc2cmd

[NLC2CMD Competition](http://nlc2cmd.us-east.mybluemix.net/)

Project to win **$2500** by taking over your frustration with bash commands to make a beautiful tool so you don't have to google it. 

# Run NL2Bash with CLAI Evaluation Framework

After cloning this repository, process data using `make data` command, and locate the pretrained model into `nl2bash/model/seq2seq(or seq2tree)` (or train a new model using the `bash-*` script and your GPU).


Move this repository into `clai/submission-code/src/submission-code/`.
It is possible to import `predict_nl2bash` using below import statement:

```python
from nlc2cmd import predict_nl2bash
```
We can also add tones of versions of `predict` functions as time goes by.

Then, evaluate and grab the (highest) score!


## Explore data

Get the data first:

 * [nl2bash](https://ibm.box.com/v/nl2bash-data)
 * [manpage](https://ibm.box.com/v/nlc2cmd-manpagedata)

and put it in `data/`. There's a Jupyter notebook that helps you. 
Well, at least I hope it helps.
