# Emotion-and-its-Stimulus-Detection
This is second project in my series of cool NLP projects/week.
I always wondered if a single model can perform two or more NLP tasks  simultaneously without having to train on every task explicitly. I learned about T5 from abhishek's talk [here](https://www.youtube.com/watch?v=4LYw_UIdd4A&t=2020s) which is a text to text transformer model trained to perform multiple tasks at once.

**Thus this week I have made a model which tries to detect emotions and also its cause from a given text. I have made use of multitasking abilities of T5:Transformer** 

After training the model I tested it on unseen ,self made data and got very good results . Below are some Examples

![Image description](https://github.com/tanulsingh/Emotion-and-its-Stimulus-Detection/blob/master/Demo/demo2.PNG)

![Image description](https://github.com/tanulsingh/Emotion-and-its-Stimulus-Detection/blob/master/Demo/New%20pro%20demo.PNG)

![Image description](https://github.com/tanulsingh/Emotion-and-its-Stimulus-Detection/blob/master/Demo/last.PNG)

# Data
The Data can be found [here](http://www.site.uottawa.ca/~diana/resources/emotion_stimulus_data/) . It conatins two files , first file 'emotions' contains emotions along with their cause and second file 'no cause' contains only the emotion and examples of text where the cause of the emotion is not present in the text.

Here are some examples from both the files:
* <'happy'> I suppose I am happy <'cause'> being so ` tiny'<\cause> ; it means I am able to surprise people with what is generally seen as my confident and outgoing personality . <\happy>
* <'happy'> This did the trick : the boys now have a more distant friendship and David is much happier . <\happy>
  
 As we can see the first sentence contains the cause of emotion but in the second sentence writer says that he is happy but does not gives the cause/reason for it.
 
 **As for my model I trained it only on the emotions data and thus it is not able to detect if there is no cause in the statement/sentence.You can surely modify the training to get this done.**
 
 # Pre-Processing
 
 T5 is a generative model i.e it generates the cause and emotions instead of predicting it , it's a little difficult to understand at first but once you do it , it's really fascinating. 
 
 If you are using only the **emotion** data then it should be pre-processed like this:
 * Original ---> <'happy'> I suppose I am happy <'cause'> being so ` tiny'<\cause> ; it means I am able to surprise people with what is generally I have used HuggingFace Library for GPT-2 Model and the whole code is written in Pytorch. I will be more than happy to share if someone takes this model and writes its equivalent in Keras/TF (that would be a good exercise) .The modelling and inference are easy to understand and self-explanatory if one reads the HuggingFace Docs.seen as my confident and outgoing personality . <\happy>
 * Input after pre-processing ---> I suppose I am happy being so ` tiny'; it means I am able to surprise people with what is generally seen as my confident and outgoing personality. + <'/s'>
 * Target ---> <'label'> happy <'cause'> being so ` tiny' + <'/s'>
  
  If you are using 'no cause' data as well then you should pre-process it like this:
  * Original --->  <'happy'>This did the trick : the boys now have a more distant friendship and David is much happier . <\happy>
  * Input ---> This did the trick : the boys now have a more distant friendship and David is much happier .
  * Target ---> <'label'> happy <'cause'> no cause found 
  
  Also note that you will have to add <'label'> and <'cause'> as tokens to the tokenizer you use for good performance
  
# Model
I have used HuggingFace Library for T5-Large Model and the whole code is written in Pytorch.The modelling and inference are easy to understand and self-explanatory if one reads the HuggingFace Docs. 

# HyperParameters

I have tested on two learning rates , the later works  better.It does not take a lot of time to train as the dataset is very small
 
| Batch_Size | MAX_LEN | EPOCHS | Learning Rate| Train Time On GPU's |
| ------------- | ------------- |------------- | ------------- | ---------|
| 1 | 50  | 5  | 3e-5  |30 mins|
| 1  | 50  | 5  | 2e-5  | 35 mins |

* Note that I have used the batch_size of one because the data had only 824 examples and it was too less , but highly recommend to get more data and not use bs of 1 , it's not good practice and model will have problems converging if done

# End Notes

* There is a lot of scope for improvement like getting more data , using the 'no cause' data for training as well , training for different batch_sizes,etc
* You can also test the model by giving two emotions in a single sentence and analyzing the results
* T5: Large was used as the data was less and with bs 1 it worked the best , but if you have a lot of data , first try to use t5small or t5 base

Best of luck Playing with the model
