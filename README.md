# Implementation of POKI(Achieving Conversational Goals with Unsupervised Post-hoc Knowledge Injection)

this repository is a simple Implementation of POKL(2022 ACL).

## Usage
This section explains steps for POKI. 

### Preprocessing: 
It includes downloading MultiWOZ dataset, performing delexicaliztion, and creating dataset for language model
```
create_dataset.sh
```

### End-to-End training:
In this step, we train SimpleTOD on the sequence of context+belief+action+lex response. 
here we train on gpt2 model (https://huggingface.co/gpt2)
```
train_end2end.sh 
```

### Generate:
use simpletod model to generate belief+action, then get context+belief+action as the input of next step.
```
generate.sh
```

### Train discrim classifier
we follow the pabst(Unsupervised Enrichment of Persona-grounded Dialog with Background Stories)
to train a discrim classifier
```
train_discrim.sh
```

### Prepare yelp corpus
you should process raw yelp reviews to the following format.

```
[If you decide to eat here, just be aware it is going to take about 2 hours from beginning to end. We have tried it multiple times, because I want to like it! I have been to it's other locations in NJ and never had a bad experience. 
,
The food is good, but it takes a very long time to come out. The waitstaff is very young, but usually pleasant. We have just had too many experiences where we spent way too long waiting. We usually opt for another diner or restaurant on the weekends, in order to be done quicker.
Family diner. Had the buffet. Eclectic assortment: a large chicken leg, fried jalapeño, tamale, two rolled grape leaves, fresh melon. All good. Lots of Mexican choices there. Also has a menu with breakfast served all day long. Friendly, attentive staff. Good place for a casual relaxed meal with no expectations. Next to the Clarion Hotel.
...
]

```

### Generate response with Post-hoc Knowledge

```
perturb.sh
```
