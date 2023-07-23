# Face Recognition Project

## Description

This project implements a face recognition system using Convolutional Neural Networks (CNNs) and Support Vector Machines (SVMs). The system can detect and recognize faces from input images and classify them based on known subjects.

## Repository Structure

- `FacialRecognition.ipynb`: The Jupyter Notebook containing the implementation of the face recognition system.
- `model/`: Directory containing the trained CNN and SVM models.\
- `pylib/`: Directory containing the functions.py.\
- `main.py`: Python file to run program.\
- `setup.py`: Python file to install dependencies.\
  
  

## Installation and Setup

To use this project, follow these steps:

1. Clone the repository:
git clone https://github.com/ayyodeji/facerecognition.git
cd face-recognition


2. Set up the Python environment:
- Make sure you have Python 3.x installed.
- Create a virtual environment (optional but recommended):
  ```
  python -m venv env
  source env/bin/activate  # For Linux/Mac
  env\Scripts\activate  # For Windows
  ```
- Install the required dependencies:
  ```
  pip install -e .
  ```

## Usage

1. Open/Run the Jupyter Notebook `FacialRecognition.ipynb` to view the implementation and documentation of the face recognition system.
2. Follow the step-by-step instructions in the notebook to train the CNN and SVM models on the provided dataset.
3. Evaluate the model performance using accuracy metrics and confusion matrices.
4. Test the face recognition system by providing input images of faces and observe the predicted subject classes.
5. Models are automatically saved in a newly created folder `model`
6. Run the program

## Sample Test Image

To test the face recognition system, you can use the provided sample test image located in the `data/test/` directory.

```python
img, expected_class = get_sample_test_image()
print(f"Expected class: {expected_class}")

authenticate(img, debug=True)
```
## Dataset available on kaggle.com `Yale faces dataset`
### For more information and detailed usage, please refer to the FacialRecognition.ipynb notebook.

Feel free to contribute to this project by opening issues or submitting pull requests!

