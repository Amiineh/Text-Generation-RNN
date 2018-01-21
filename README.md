# Text-Generation-RNN
In this project, we use a Recursive Neural Network with LSTM cells to generate text. We use a character-based dictionary to predict the next character at each time step.
During training time, the input of the network is a window of 40 characters, in a one-hot vector with the same length of our dictionary, and stride 3. And we use the next character (41st character) as the target character.
During test time, we pick a random window of our training text and concatenate the predicted output with the input and continue generating the next character.

# Results
Some of the results are shown here. For more results, you can take a look at results.txt. 


Epoch 1
------
Input: 
```
and boundlessly foolish naivete is invol
```
Generated text: 
```
  iaa a ga  heu  esaid      e   eyaux  ia   tae ,s   e   t al 9  e,a e     ae ei  e  a  rnan s aaa  �eea  a de  t  ee  � hie ae  3ai eoyhe a  eeee e at d   xnirale2 ha   a hhlai)
 e  6   j ta tee
 i  eioae a aaa aea ehz   jee ke e  lwaw ent  isi nn aojet c ee ha  s  e   a  a   si 2  h
aa ala ti eiref oai i e   aa4 ia caoea   i   !e   eh    a i   a aien aa     etta     ty  p  3e .ee at  s a     l k
```


Epoch 3
------
Input: 
```
can be! i know of nothing more stinging
```
Generated text: 
```
of the racicing and wostins the afe in ere and ins moferese and ire and
fatile fore ist and and ande sees in the soule
--is sing  atiche icis of ilt of and ans and chate and sicins and sesthe are itis int wice the ware alitincos wathe goure int al allienthe mint ande ist in thes of ste of isterealis we he the apen all se thepreathe and are ind the athe ance with if the fore of alle whe ith inthout
```


Epoch 5
------
Input: 
```
lism of art, perhaps as
music, or as lov
```
Generated text: 
```
en which is indictloman which
spective of precing and such a manton the greation of cals celspation of all and furtical constiently spenition of pressplical of the sperious the spech seans of calsens of the with if in the sperion of precess the spring which
spectations with exicht presprecess, a sence of the mpansently and with even soelifice such all and
fere pracise are spensing with
somethe can
```


Epoch 48
------
Input: 
```
to have legitimate rights in science. in
```

Generated text: 

``` the sense in their distrust is the
"pridrinatical specticises and flattered and nothing with a list
gregterable as the same to the restener, and thereofyent of philosopher of serveat constrained, and with a disting the signly and sometime to nature" is not the sometimes conscience if and distrust is an activity be the will" with a different in which with a discipling to themself with a llogis lig
```

