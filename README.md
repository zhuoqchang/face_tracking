## Installation

### Virtual Environment
Since we are dealing with Python code, it would be ideal to set up a python virtual environment first. I recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and Python 3.6. Instructions can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Dependencies
Open the terminal and install the following dependencies
1. Dlib  
 `cd ~ && git clone https://github.com/davisking/dlib.git`  
 `cd dlib && python setup.py install`

2. Face recognition library  
 `pip install face_recognition`

3. Opencv python  
 `pip install opencv-python`

## Usage
Put the videos in _input_ folder and then run  
`python detect_face.py -i input/ -o output/`
