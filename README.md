
#INF5860 Oblig 1
In this assignment you will practice building a image classification algorithm. You will first implement a softmax classifier and move on to
building a simple modular *framework for neural networks*. The goals of this assignment are as follows:

- Understand training, validation and test **splits** and the use of validation data for **hyperparameter tuning**
- Implement and train a **Softmax** classifier
- Implement and train a **Two layer neural network** classifier
- Get a hands-on experience of how **heigher-level representations** can help build **invariance** and boost performance.
- Understand how neural networks is arranged in layered architectures
- How **backpropagation** work and how to implement it
- Implement the **gradient decent** update rule.


##The Parts to Complete
#### Part 1: Implement a Softmax classifier
The Jupyter Notebook softmax.ipynb will walk you through implementing the Softmax classifier.

#### Part 2: Two-Layer Neural Network 
The Jupyter notebook two_layer_net.ipynb will walk you through the implementation of a two-layer neural network classifier.

#### Part 3: Image Features
The IPython Notebook features.ipynb will walk you through this exercise, in which you will examine
the improvements gained by using higher-level representations as opposed to using raw pixel values.
 
#### Part 4: Backpropagation
The Jupyter notebook `backpropagation.ipynb` will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth.

## Get Started
Download the starter code from git with:

    git clone https://github.com/anneschistad/INF5860_Oblig1.git

**[Option 1] Use IFI-linux computer:**

If you have not installed Jupyter on you IFI-computer, you can run it with:

    /hom/sigmunjr/anaconda/bin/jupyter notebook

Instead of just `jupyter notebook`. Otherwise everything **should** work fine.

**[Option 2] Use Anaconda:**
The preferred approach for installing all the assignment dependencies is to use
[Anaconda](https://www.continuum.io/downloads), which is a Python distribution
that includes many of the most popular Python packages for science, math,
engineering and data analysis. Once you install it you can skip all mentions of
requirements and you are ready to go directly to working on the assignment.

**[Option 3] Manual install, virtual environment:**
If you do not want to use Anaconda and want to go with a more manual and risky
installation route you will likely want to create a
[virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)
for the project. If you choose not to use a virtual environment, it is up to you
to make sure that all dependencies for the code are installed globally on your
machine. To set up a virtual environment, run the following:

```bash
cd assignment2
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
```

**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the `assignment2` directory:

```bash
cd code/datasets
./get_datasets.sh
```


**Start Jupyter:**
After you have the CIFAR-10 data, you should start the **jupyter notebook** server
from the `INF5860_Oblig1` directory. In the `INF5860_Oblig1` directory run:

    jupyter notebook
    
Then you will se the list of the notebooks, that should be solved in the provided order: **softmax.ipynb**, **two_layer_net.ipynb**, **features.ipynb** and **backpropagation_network.ipynb**.
By clicking on a notebook, you can start working on an assignment. You can run a cell with `shift+enter` or `ctrl+enter`. You can find more
keyboard shortcuts [here](https://www.cheatography.com/weidadeyue/cheat-sheets/jupyter-notebook/).

If you are unfamiliar with jupyter, you can test it out with [try Jupyter](https://try.jupyter.org/). There you can find a simple overview
in the `Welcome to Python.ipynb` notebook. To get a more extensive guide you can go to `communities/pyladies/Python 101.ipynb`.

**NOTE:** If you are working in a virtual environment on OSX, you may encounter
errors with matplotlib due to the
[issues described here](http://matplotlib.org/faq/virtualenv_faq.html).
You can work around this issue by starting the IPython server using the
`start_ipython_osx.sh` script from the `assignment2` directory; the script
assumes that your virtual environment is named `.env`.


### Submitting your work:
Once you are done working, run the `collectSubmission.sh` script; this will produce a file called
`INF5860_Oblig1.zip`. 

Then upload the zip-file file to [devilry](devilry.ifi.uio.no) (devilry.ifi.uio.no). You can make multiple submissions before the deadline.

Good luck!
