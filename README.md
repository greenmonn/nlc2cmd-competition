# nlc2cmd

[NLC2CMD Competition](http://nlc2cmd.us-east.mybluemix.net/)

Begone, frustration over bash commands! This is a tool that infers bash commands, so you don't have to google them.

Doesn't hurt that we'll win **$2500** along the way.

# Run NL2Bash with CLAI Evaluation Framework

After cloning this repository, process data using `make data` command, and load any pretrained models into `nl2bash/model/seq2seq(or seq2tree)`. (You may also train a new model using the `bash-*` script and your GPU).

Move this repository into `clai/submission-code/src/submission_code/`.
It is possible to import `predict_nl2bash` using the below import statement:

```
python
from nlc2cmd import predict_nl2bash
```
We can also add tons of versions of `predict` functions as time goes by.

Then, evaluate and grab the (highest) score!


## Explore data

Get the data first:

 * [nl2bash](https://ibm.box.com/v/nl2bash-data)
 * [manpage](https://ibm.box.com/v/nlc2cmd-manpagedata)

and put it in `data/`. There's a Jupyter notebook that helps you. 
