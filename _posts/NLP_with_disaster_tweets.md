---
title: "Natural Language Processing with Disaster Tweets: Kaggle Competition Attempt"
date: 2020-12-30
tags: [nlp, classification, wikinet-103, deep learning, rnn, natural language processing, lstm, awd, python]
header:
excerpt: "Natural Language Processing with Disaster Tweets"
mathjax: "true"
---

# Natural Language Processing with Disaster Tweets: Kaggle Competition Attempt



In this article, I attempt the Natural Language Processing with Disaster Tweets Kaggle competition. The competition entails predicting which tweets are about real disaster, and which are not. If we are able to predict whether a tweet is about a real disaster or emergency (e.g. a tweet from someone witnessing a wildfire), the relevant agencies can be contacted immidiatley to provide assistance in dealing with the disaster. The full competition description is available on the Kaggle website.



I will be using the knowledge I have recently gained from the course I completed called *Practical Deep Learnign for Coders* by fastai. The course introduces you to many useful applications of deep learning today and how to solve deep learning problems using the fastai and PyTorch libraries. You will also learn about ethical issues machine learning practioners face today and how to develop frameworks to tackle these issues. It is truly an incredible course and I highly recommend it. See https://course.fast.ai/ for more details.



I will tackle this text classification problem using the Universal Language Model Fine-tuning (ULMFit) approach, which is outlined in chapter 10 of *Practical Deep Learnign for Coders*. The approach consists of three steps:



1. Start by fine-tuning a pretrained language model on the target corpus of text in order to learn useful embeddings for the words in the corpus.



2. Fine-tune the model obtain after completing step 1 to a text classifier.



3. Use the text classifier obtained in step 2 to predict whether a tweet pertains to a disaster or not.



I will be using the AWD-LSTM architecture for my recurrent neural network (RNN) language model. The langauge model has already been trained on the WikiText-103 dataset and has learned useful embeddings from there. Even though an accurate text classifier can be built from the embeddings in the pretrained model, we can build an even more accurate classifier by fine-tuning the pretrained model on the target corpus. The reason for this is that the language style used in tweets is, in general, different from that of Wikipedia articles. By allowing the langauge model to learn the style of the target corpus, the emdeddings learnt will be even more useful.

Let's begin by importing the necessary libraries and the training and test datasets.


```
!pip install -Uqq fastbook

import fastbook

fastbook.setup_book()
```


```
from fastai.text.all import *



import pandas as pd
```


```
# Import the training and test set

train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")
```

Now, we take a look at the heads of the training and test sets to get a feel for what they contain.


```
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation orders in California</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, stay safe everyone.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are fleeing across the street, I cannot save them all</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>



I will only be using the tweets in the "text" column as predictors in my final classification model. I'm doing this because I would like to focus on building a pure text classifier.



Next, we move on to fine-tuning our pretrained language model. Since we are not yet looking to predict the labels associated with the tweets, and are only looking to learn the best possible embeddings for the words within the tweets, we use the full corpus (both the training and test sets) to train our language model.



We are not commiting data leakage as the labels of the tweets have not been used in any way to influence the models predictions.

The first step here is to concatentate the "text" columns in the training and test sets.


```
# Concatenate the "text" in both the training and test sets to train the language model

frames = [train_df["text"], test_df["text"]]

lm_df = pd.DataFrame(pd.concat(frames, axis=0))
```


```
lm_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Forest fire near La Ronge Sask. Canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13,000 people receive #wildfires evacuation orders in California</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school</td>
    </tr>
  </tbody>
</table>
</div>



Now, we create the `Dataloaders` for training the language model. fastai automatically takes care of tokenization and numericalization when creating the `DataBlock`. Using `is_lm=True` tells `TextBlock` that we want to train a language model and that it should create the response accordingly. `Dataloaders` also takes care of batch collation (which can be tricky in NLP problems).



We use a validation set comprising of 10% of the input data.


```
# Create the dataloaders for the language model

dls_lm = DataBlock(blocks=TextBlock.from_df('text', is_lm=True),

                    get_x=ColReader('text'),

                    splitter=RandomSplitter(valid_pct=0.1, seed=42)).dataloaders(lm_df, bs=128, seq_len=80)
```





    /usr/local/lib/python3.6/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      return array(a, dtype, copy=False, order=order)
    

Now, we have a look at the tokenized versions of the tweets. We observe that a couple of extra words are now included that begin with 'xx'. These are there to help the language model identify, for example, the start of a new tweet ('xxbos'), or a capitalization ('xxmaj').


```
dls_lm.show_batch(max_n=2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>text_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj saving the xxmaj city in xxmaj old xxmaj town : xxmaj the xxmaj proposed xxmaj demolition of xxunk xxmaj west xxmaj xxunk http : / / t.co / xxunk xxunk for xxunk xxbos xxmaj great xxmaj british xxmaj bake xxmaj off 's back and xxmaj dorret 's chocolate gateau collapsed - xxup jan xxup xxunk http : / / t.co / xxunk http : / / t.co / xxunk xxbos xxunk earthquake occurred 5 km s of xxmaj</td>
      <td>xxmaj saving the xxmaj city in xxmaj old xxmaj town : xxmaj the xxmaj proposed xxmaj demolition of xxunk xxmaj west xxmaj xxunk http : / / t.co / xxunk xxunk for xxunk xxbos xxmaj great xxmaj british xxmaj bake xxmaj off 's back and xxmaj dorret 's chocolate gateau collapsed - xxup jan xxup xxunk http : / / t.co / xxunk http : / / t.co / xxunk xxbos xxunk earthquake occurred 5 km s of xxmaj volcano</td>
    </tr>
    <tr>
      <th>1</th>
      <td>t.co / xxunk # earthquake xxbos xxunk fire xxbos # dating xxmaj absolute xxmaj approaching : xxmaj unique xxunk course program to obliterate approach anxiety to get more dates . http : / / t.co / xxunk xxbos xxunk xxunk i usually never try to express xxunk for fear of the hate xxunk xxbos xxmaj dr . xxmaj xxunk on # wildfire management : xxunk and size of fires areas affected and costs of fighting them all show xxunk xxunk</td>
      <td>/ xxunk # earthquake xxbos xxunk fire xxbos # dating xxmaj absolute xxmaj approaching : xxmaj unique xxunk course program to obliterate approach anxiety to get more dates . http : / / t.co / xxunk xxbos xxunk xxunk i usually never try to express xxunk for fear of the hate xxunk xxbos xxmaj dr . xxmaj xxunk on # wildfire management : xxunk and size of fires areas affected and costs of fighting them all show xxunk xxunk #</td>
    </tr>
  </tbody>
</table>


We now create our `Learner`, which encompasses our pretrained model.


```
# Intantiate the language model

learn = language_model_learner(

    dls_lm, AWD_LSTM, drop_mult=0.3, 

    metrics=[accuracy, Perplexity()]).to_fp16()
```

Now, we find the optimal maximum learning rate to use to train the head of the pretrained model. fastai automatically uses discriminative learning rates in a transfer learning problem to achieve maximum validation set accuracy. This involves using a lower learning rate for earlier layers (i.e. layers that contain the pretrained weights), and use higher weights for later layers (including the head).


```
learn.lr_find()
```








    SuggestedLRs(lr_min=0.13182567358016967, lr_steep=0.14454397559165955)




![png](NLP_with_disaster_tweets_files/NLP_with_disaster_tweets_19_2.png)


Here we see that the learning rate that results in the steepest loss reduction is around 1e-1. Using this learning rate as `lr_max`, we fit for one epoch using `fit_one_cycle`. We save the model after this training cycle.


```
# Fine-tune the pretrained model

# Fit for one epoch and then save the model

learn.fit_one_cycle(1, lr_max=1e-1)

learn.save('1epoch')
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.990792</td>
      <td>3.373023</td>
      <td>0.431199</td>
      <td>29.166559</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>





    Path('models/1epoch.pth')



Next, we unfreeze the full RNN and find the optimal maximum learning rate for training.


```
# Unfreeze the pretrained model and find the optimal max learning rate

learn.unfreeze()

learn.lr_find()
```








    SuggestedLRs(lr_min=0.0007585775572806596, lr_steep=0.001737800776027143)




![png](NLP_with_disaster_tweets_files/NLP_with_disaster_tweets_23_2.png)


Here we see that it is 1e-3.


```
learn.fit_one_cycle(6, 1e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.141677</td>
      <td>3.320533</td>
      <td>0.444605</td>
      <td>27.675098</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.130927</td>
      <td>3.274940</td>
      <td>0.452928</td>
      <td>26.441652</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.075932</td>
      <td>3.206075</td>
      <td>0.469497</td>
      <td>24.682030</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.027065</td>
      <td>3.201564</td>
      <td>0.475896</td>
      <td>24.570919</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.962674</td>
      <td>3.201890</td>
      <td>0.481616</td>
      <td>24.578947</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.897123</td>
      <td>3.201150</td>
      <td>0.482484</td>
      <td>24.560757</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>


We now save the encoder of the language model (i.e. the model without the head) to use for text classification.


```
# Save the encoder of the language model for use in text classification

learn.save_encoder('finetuned')
```

Next, we move on to building the text classifier. We must start by building the `Dataloaders` and `DataBlock`. Now, we use the binary response as our target and the tweets as our predictors. We use a validation set of 20% of the original training set.


```
# Create the classifier Dataloaders

dls_clas = DataBlock(

    blocks=(TextBlock.from_df("text", vocab=dls_lm.vocab),CategoryBlock),

    get_y = ColReader("target"),

    get_x = ColReader("text"),

    splitter=RandomSplitter(seed=42, valid_pct=0.1)

).dataloaders(train_df, bs=128, seq_len=72)
```





    /usr/local/lib/python3.6/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      return array(a, dtype, copy=False, order=order)
    


```
dls_clas.show_batch(max_n=3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos _ \n▁ xxrep 5 ? xxup retweet \n▁ xxrep 7 ? \n▁ xxrep 5 ? xxup follow xxup all xxup who xxup rt \n▁ xxrep 7 ? \n▁ xxrep 5 ? xxup xxunk \n▁ xxrep 7 ? \n▁ xxrep 5 ? xxup gain xxup with \n▁ xxrep 7 ? \n▁ xxrep 5 ? xxup follow ? xxunk # xxup xxunk \n▁ # xxup ty</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos . : . : . : . : . : . : . : . : . : . : . : . : . : . : . : . : . : . : . : . : . : xxup rt xxunk : # xxunk \n\n xxmaj indian xxmaj army xxunk _ http : / / t.co / xxunk g</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxbos xxup info xxup s. xxup wnd : xxunk / 6 . xxup xxunk : xxup xxunk xxup xxunk . xxup exp xxup inst xxup apch . xxup rwy 05 . xxup curfew xxup in xxup oper xxup until 2030 xxup z. xxup taxiways xxup foxtrot 5 &amp; &amp; xxup foxtrot 6 xxup navbl . xxup tmp : 10 .</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


Here we create the text classifier model and load the encoder from our trained language model. 


```
# Create the tweet classifier

learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, 

                                metrics=[accuracy, F1Score()]).to_fp16()

# Load the encoder obtained from the language model

learn = learn.load_encoder('finetuned')
```

Now, we train the head of the model.


```
learn.fit_one_cycle(1, 2e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>f1_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.936312</td>
      <td>0.496844</td>
      <td>0.759527</td>
      <td>0.682842</td>
      <td>00:08</td>
    </tr>
  </tbody>
</table>


Using gradual unfreezing, we train the full model. We also pass a `slice` to `lr_max`. The first value in the slice is the learning rate for earlier layers, and the last value in the slice is the learning rate for later layers.


```
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>f1_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.664666</td>
      <td>0.466251</td>
      <td>0.784494</td>
      <td>0.721088</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>



```
learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>f1_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.532604</td>
      <td>0.463982</td>
      <td>0.785808</td>
      <td>0.730579</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>



```
learn.unfreeze()

learn.fit_one_cycle(3, slice(1e-3/(2.6**4),1e-3))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>f1_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.472363</td>
      <td>0.465664</td>
      <td>0.789750</td>
      <td>0.741100</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.458194</td>
      <td>0.467102</td>
      <td>0.787122</td>
      <td>0.723549</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.436414</td>
      <td>0.463984</td>
      <td>0.787122</td>
      <td>0.730897</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>


We see that we obtain a text classifier with an 78.7122% accuracy and an F1 score of 0.7309. Finally, we obtain the predictions on the test set and export the csv.


```
test_dl = dls_clas.test_dl(test_df)

preds = learn.get_preds(dl=test_dl)
```






```
# Convert the predictions to a list of 0s and 1s

preds = [int(preds[0][i][1] > 0.5) for i in range(0, len(preds[0]))]



# Create the predictions dataframe and export

predictions = pd.DataFrame({"id":test_df["id"], "target":preds})



predictions.to_csv("predictions.csv")
```

According to Kaggle, my submission had an F1 score of 0.80079. It is truly amazing to see what is possible with fastai and PyTorch with so few lines of code. 



Possible areas of improvement include: attempting to utilise the other features available in the data sets to predict the response (the missing values would require imputation), and utilising ensemble learning with other models (such as a naive Bayes classifier).
