# Emotion-and-its-Stimulus-Detection
This is second project in my series of cool NLP projects/week.
I always wondered if a single model can perform two or more NLP tasks  simultaneously without having to train on every task explicitly. I learned about T5 from abhishek's talk [here](https://www.youtube.com/watch?v=4LYw_UIdd4A&t=2020s) which is a text to text transformer model trained to perform multiple tasks at once.<br>
Thus this week I have made a model which tries to detect emotions and also its cause from a given text. I have made use of multitasking abilities of T5:Transformer 

After training the model I tested it on unseen ,self made data and got very good results
