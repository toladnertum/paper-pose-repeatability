
# Repeatability Package

This folder contains a repeatability package for:

- Paper: `Perception with Guarantees: Certified Pose Estimation via Reachability Analysis`
- Venue: `under review`

## Folder structure


- `./`                      : base path                             
  - `./code`                : path to code                          
    - `./cora`              : path to [CORA](https://cora.in.tum.de/)
    - `./scripts`           : path to auxiliary Matlab scripts                 
    - `./main.m`            : **main Matlab script**
  - `./data`                : path to data                  
  - `./results/<evalname>`  : path to results (created after execution)
    - `./evaluation`        : path to store any evaluation results  
    - `./plots`             : path to plots    
    - `./results.txt`       : logs of all outputs to command window    
  - `./Dockerfile`          : Dockerfile
  - `./license.lic`         : place [license file](#step-1-installation) here
  - `./README.md`           : read me file (this file)
  - `./run.sh`              : **main script** to run [from command line using Docker](#run-from-command-line-recommended)
  - `./settings.sh`         : settings for bash scripts
  - `./screen.sh`           : script to run `run.sh` [within a linux screen](#run-from-command-line-recommended)

## Scope

- The repeatability package runs the synthetic experiment using target `stripes` per default to reduce evaluation time. 
  This recomputes Tab. 1 for `stripes` and Tab. 3 for `200x200`, which can be found at the end of `./results/<evalname>/results.txt`
- Estimated time: `30min`
- Please check the settings in Sec. 3 in `main.m` for alternatives: other targets (Tab. 1), real-world experiment (Tab. 2).

To run the repeatability package, please follow these steps:
- [Step 1: Installation](#step-1-installation) *(Docker or Matlab)*
- [Step 2: Run the code](#step-2-run-the-code) *(Docker or Matlab)*
- [Step 3: View results](#step-3-view-results)

## Step 1: Installation
This folder contains the code as well as a Docker file to run the code in one click.

However, you need to provide a Matlab license.
You can either specify 

- a) a [license server](#a-license-server-recommended) (recommended), 
- b) a [license file](#b-license-file), or 
- c) [run the repeatability package directly in Matlab](#c-install-matlab-and-required-toolboxes-not-recommended):

### a) License server (recommended)

1. Ask your Matlab administrator if a Matlab license server is available.
2. In `settings.sh`, configure the license server : `LICENSE_SERVER=<port>@<hostname>`. 

➡️ Proceed with [Step 2a: Run from command line using Docker](#a-run-from-command-line-using-docker-recommended)

### b) License file
Download a license file `license.lic` to run the code:
1. Create a Matlab license file: 
	For the Docker container to run Matlab, one has to create a new license file for the container.
	Log in with your Matlab account at https://www.mathworks.com/licensecenter/licenses/.
	Click on your license, and then navigate to
	1. "Install and Activate"
    1. "View activated computers"
	1. "Activate a Computer"
	(...may differ depending on how your licensing is set up).
2. Choose:
	- Release: `R2024b`
	- Operating System: `Linux`
	- Host ID: `0242AC11000a` (= Default MAC of Docker container)
	- Computer Login Name: `matlab`
	- Activation Label: `<any name>`
3. When prompted if the software is already installed, choose "Yes".
4. Download the file and place it next to `Dockerfile`.

➡️ Proceed with [Step 2a: Run from command line using Docker](#a-run-from-command-line-using-docker-recommended).

### c) Install Matlab and required toolboxes (not recommended)

Install Matlab on your system and install all required toolboxes for CORA (see Sec. 1.3 in [CORA manual](https://cora.in.tum.de/manual)). The CORA repository is already included in `./code/cora`, so you don't have to clone it.

➡️ Proceed with [Step 2b: Run from Matlab](#b-run-from-matlab).


## Step 2: Run the code

Run the code depending on the chosen option in [step 1](#step-1-installation):
- [a) Run from command line using Docker](#a-run-from-command-line-using-docker-recommended) (recommended).
- [b) Run from Matlab](#b-run-from-matlab)

---

### a) Run from command line using Docker (recommended)

Make all bash scripts executable using 

    chmod +x *.sh

(see also [bug fix: windows/linux line breaks](#known-error-messages) below).

#### i) Run the package within a linux screen (recommended)

If you are using [linux screens](https://www.howtogeek.com/662422/how-to-use-linuxs-screen-command/),
you can run the package in one click in a Docker container using

    ./screen.sh <evalname> <gpu-device>

where the argument `<evalname>` is used to name the evaluation run (defaults to datetime),
and the optional argument `<gpu-device>` is used to select the GPU (see [GPU settings](#gpu-settings) below).

Linux screens are helpful when running the script on a server to ensure it finishes correctly even if your connection is interrupted.
You can always detach from the screen using `CTRL+A+D` and reattach using 

    screen -rd $SCREEN_NAME

where `SCREEN_NAME` is as in `settings.sh` (see [variables](#variables) below) or using `screen -ls`.

➡️ Proceed with [Step 3: View results](#step-3-view-results)

#### ii) Run the package directly

You can also run the evaluation in one click in a Docker container using the `run.sh` script:

	./run.sh <evalname> <gpu-device>

where the arguments `<evalname>` and `<gpu-device>` are as above.

➡️ Proceed with [Step 3: View results](#step-3-view-results)

#### Variables

To set the variables `DOCKER_NAME` and `SCREEN_NAME` automatically, you can call

    source settings.sh <evalname>

which makes the variables available in the current terminal instance,
where `<evalname>` is again the name of the evaluation.

For example, you can then stop a run using

    docker stop $DOCKER_NAME

#### GPU Settings

For Docker to use the GPU, you have to specify the `<gpu-device>` Docker should use.
You can find your available GPUs using the command `nvidia-smi`.
Possible options are the GPU id (e.g., `0`), `all`, and `none` (default).
Read more about it here: https://docs.docker.com/desktop/features/gpu/.

Please note that this setting might not be necessary for this repeatability package.

---

### b) Run from Matlab

Alternatively, open this directory in Matlab and run:

	addpath(genpath('./code')); 
    main('<evalname>');

where the optional argument `<evalname>` is used to name the evaluation run (defaults to datetime).
The results will be stored to `./results/<evalname>`.
	
**Note:** Please ensure that all required toolboxes for CORA are installed (see [Step 1c:](#c-install-matlab-and-required-toolboxes-not-recommended) above).

➡️ Proceed with [Step 3: View results](#step-3-view-results)

---

## Step 3: View results
	
The results will be stored to `./results/<evalname>`.

If you run the repeatability package from the command line within Docker,
you can view intermediate results by copying the current `results` folder out of the Docker container using

    docker cp "$DOCKER_NAME":/results .

where `DOCKER_NAME` is as in `settings.sh` (see [variables](#variables) above) or using `Docker ps`.


## Known error messages

- If running `run.sh`/`screen.sh` results in obscure error messages (`$'\r': command not found`), 
  it might be due to different line breaks in `run.sh`/`screen.sh` using windows/linux. 
  You can fix it using:

      sed -i 's/\r$//' *.sh

- When running the evaluation in Docker, Docker might randomly stop (with message "Killed.") if not enough memory is available.
