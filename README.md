## A Python library for working with isometries and orthogonal transforms.

The code in this repository started from trying to tackle 
a  [question on the composition of two rotations](https://math.stackexchange.com/q/4999941/435819) 
(in 2-D and 3-D).
The code has been developed to represent isometries and orthogonal transformations
when solving problems involving rotations, reflections, etc., in 2-D and 3-D.

The emphasis is on using _direct geometric methods_
to model rotations, etc., 

Using matrices is an alternative way of modelling isometries
and these are used partially, mainly for tasks such as checking
that two isometries are 'equal'. Modelling orthogonal
rotations with quaternions is done only briefly as an exercise.

See the folders `examples` and `tests` for illustrations on 
the classes that are used to model orthogonal transformations
and isometries.


## Installation

The code has been tested using python 3.12.

Make an environment:

    python3.12 -m venv my_env

Activate:

    source my_env/bin/activate

Upgrade pip:

    python -m pip install --upgrade pip

Install requirements:

    python -m pip install -r requirements.txt

Install this library:

    python -m pip install .

or, for development mode:

    python -m pip install -e .

