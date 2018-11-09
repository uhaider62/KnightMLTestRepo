# Brain installation instructions #
NOTE: OpenCV version less than 3 should be used
1. Begin by installing Python 3.5.3 and add its installation directory to path. Package manager pip comes pre-installed for Python 3 (OpenCV is part of its packages).

2. Install necessary packages by copy-pasting the following into your terminal
    ```
    python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose opencv-contrib-python imutils
	
    python -m pip install --upgrade tensorflow
	
    python -m pip install scikit-learn
    python -m pip install pillow
    python -m pip install h5py
	pip install keras
	
    ```
Note if you python 2 installed previously use `py` instead of `python`.

3. Run the provided sample scripts and make sure you do not see any errors in console.

