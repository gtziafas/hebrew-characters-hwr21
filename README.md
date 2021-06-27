# hebrew-characters-hwr21
Code for Hebrew character and style recognition for the Dead See Scrolls dataset, in the context of the "Handwritting Recognition" course of RUG, 2021.

## Installing dependencies
Project works with Python 3.8. To install the dependencies run the following commands in the project root.
```bash
python3 -m venv venv
. venv/bin/activate
pip install -r REQUIREMENTS.txt
```

## Pipeline
To run the pipeline, use 

```bash
python3 main.py /path/to/test/images/
```

## Testing components independently

Most of the components of the pipeline can be tested independently. To run them, see the comment at the end of each Python script.

Example, found at the end of `line_segm_astar.py`:
```python
# command:
# python3 -m qumran_seagulls.preprocess.line_segm.line_segm_astar data/images/P106-Fg002-R-C01-R01-binarized.jpg
# run from project root folder
```