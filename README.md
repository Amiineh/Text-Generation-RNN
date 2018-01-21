# Text-Generation-RNN
In this project, we use a Recursive Neural Network with LSTM cells to generate text. We use a character-based dictionary to predict the next character at each time step.
During training time, the input of the network is a window of 40 characters, in a one-hot vector with the same length of our dictionary, and stride 3. And we use the next character (41st character) as the target character.
During test time, we pick a random window of our training text and concatenate the predicted output with the input and continue generating the next character.

## Results
Some of the results are shown here. For more results, you can take a look at results.txt. 

##### Epoch 1
```loss = 3.449968```
Input: 
```
"and boundlessly
foolish naivete is invol"
```

Generated text: 
```
"  iaa a ga  heu  esaid      e   eyaux  ia   tae ,s   e   t al 9  e,a e     ae ei  e  a  rnan s aaa  �eea  a de  t  ee  � hie ae  3ai eoyhe a  eeee e at d   xnirale2 ha   a hhlai)
 e  6   j ta tee
 i  eioae a aaa aea ehz   jee ke e  lwaw ent  isi nn aojet c ee ha  s  e   a  a   si 2  h
aa ala ti eiref oai i e   aa4 ia caoea   i   !e   eh    a i   a aien aa     etta     ty  p  3e .ee at  s a     l k"
```
