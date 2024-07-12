### The Trained Model
The machine vision model can be downloaded from the following link:
https://drive.google.com/drive/folders/1dO6uGhXzslopCXj0BiIqxUK4FJsW8ziI?usp=sharing

Please contact will.bowman@uci.edu if the above link does not work.

The model should be placed in the "sample_vision_processing_pipeline" directory and named "Machine_Vision.model".

### User Guide for Script Series

This user guide provides detailed instructions for setting up and running a series of scripts on an Ubuntu system, particularly focused on a hierarchical CRF model with an optional human-in-the-loop (HITL) approach.

#### Steps

1. **Update and Upgrade Ubuntu**:
    - Open a Terminal.
    - Run `sudo apt-get update` to update the package lists.
    - Then, `sudo apt-get upgrade` to upgrade all your installed packages to their latest versions.

2. **Install Python 3.8.13**:
    - First, add the deadsnakes PPA to get access to the specific Python version: `sudo add-apt-repository ppa:deadsnakes/ppa`
    - Update package lists again: `sudo apt-get update`
    - Install Python 3.8.13: `sudo apt-get install python3.8`
    - Verify the installation: `python3.8 --version`

3. **Install Pip**:
    - Ensure pip is installed for Python 3.8: `sudo apt-get install python3.8-pip`
    - Verify pip installation: `pip3 --version`

4. **Create a Virtual Environment**:
    - Install the virtual environment package: `sudo apt-get install python3.8-venv`
    - Create a new virtual environment: `python3.8 -m venv HierarchicalCRF`
    - Activate the virtual environment: `source HierarchicalCRF/bin/activate`

5. **Install Dependencies**:
    - Ensure you have a 'requirements.txt' file in your current directory.
    - Install required Python packages: `pip install -r requirements.txt`
    - Install pydensecrf library: `pip install git+https://github.com/lucasb-eyer/pydensecrf.git`

6. **Run the Bash Script**:
    - Follow the instructions below to run the bash script that manages the process flow, including optional HITL steps.

    1. Format the script in unix style: `dos2unix run_process.sh`
	2. Make the script executable: `chmod +x run_process.sh`
	3. Run the script: `./run_process.sh`
	4. Follow the on-screen prompts to proceed through the workflow.

This guide and script facilitate a streamlined process, from system preparation to the execution of a complex series of Python scripts, optionally incorporating human judgment via the HITL approach.
