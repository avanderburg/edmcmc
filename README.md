# edmcmc
<H3>Easy Differential-Evolution Markov Chain Monte Carlo (EDMCMC) in Python</H3>
![EDMCMC](https://media1.giphy.com/media/IMe5BBRuJD6EL1zC2O/giphy.gif)

<H3> How to Install </H3>

<b>Dependencies: </b>
<ul>
  <li> numpy (tested on version 1.20.1, but no known compatibility issues) </li>
</ul>
  
<b>Installation:</b>

<ol>
  <li> Download <a href='https://github.com/avanderburg/edmcmc/blob/main/edmcmc.py'> edmcmc.py</a> and save it somewhere on your drive temporarily. In this example, I will assume it was saved to the "Downloads" folder on a unix machine (located at <code>~/Downloads/</code> </li>
  <li> Open a python terminal and run the following commands: 
  <ul>
    <li><code> import sys </code></li>
    <li> <code> print(sys.path) </code></li>
  </ul>
  
  On my machine, that returns the following output: 
   <code>['', '/anaconda3/lib/python37.zip', '/anaconda3/lib/python3.7', '/anaconda3/lib/python3.7/lib-dynload', '/anaconda3/lib/python3.7/site-packages', '/anaconda3/lib/python3.7/site-packages/aeosa', '/anaconda3/lib/python3.7/site-packages/mpyfit-0.9.0-py3.7-macosx-10.7-x86_64.egg']</code>
  This is a list of directories where Python will search for files to import when asked. 
  </li>
  
  <li> Move edmcmc.py into one of the directories listed in the previous step. Since <code> '/anaconda3/lib/python3.7/site-packages'</code> is where other third-party software is usually kept, I chose to put it in there. This can be done with a GUI file system viewer, or using the following command: <code>mv ~/Downloads/edmcmc.py /anaconda3/lib/python3.7/site-packages/</code>
  </li>
  </ol>


At this point, edmcmc.py should be installed. Run python, and test it using <code>import edmcmc</code> . If that runs successfully, you can use the <a href='https://github.com/avanderburg/edmcmc/blob/main/test_edmcmc.ipynb'>online test notebook</a> to fully test the functionality. Note that multi-processing does not work on all machines yet. 

  
