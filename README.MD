# Machine Boom Lowest Point Prediction
## Project Documentation
Beside this README file the project documentation is stored in Overleaf --> https://www.overleaf.com/read/ftppxzzkjsmf

## Docker Interface
### Mounted volume folders
Inside the docker container there are several mounted directories:
|volume|mounted folder|comment|
|--|--|--|
| `/mnt/simulation` | `src/simulation/` | Contains the hose simulation |
| `/mnt/models` | `models/` | Saved models storage |
| `/mnt/notebooks` | `src/notebooks/` | Jupyter notebook collection |
| `/mnt/data` | `data/` | Destination of all simulation exports |

Volumes are specified in `bin/container-start`.

### Build an image
The bash script runs based on the `.env` arguements.
```bash
$ ./bin/build-image
```
However, the script passes these arguments to `docker build`. Image tag will be `machine-boom-project` after all.

#### Requirements before build an image
- Place a license file in the project root folder. Note: Make sure each license file has the correct extension. Each `.lic` file will be copied into the container
- Specify agx version and distribution in `.env`. All Versions are listed at Algoryx [Changelog](https://www.algoryx.se/documentation/complete/agx/tags/latest/doc/UserManual/source/changelog.html) on the website.

### Run a container
By default, the container starts detached in background. **Note:** Do not start directly with `docker run`, 
some environment variables will be missing. 
```bash
$ ./bin/container-start [<docker run args>]
```

The Image based on a `nvidia/cuda` on `ubuntu`. However, to accelerate and use gpu capacity add the gpu flag:
```bash
$ ./bin/container-start --gpus all
```

#### Exposed ports
A running container instance offers a number of services through exposed ports:

|Inner Port|Exposed Port|Service|Comment|
|--|--|--|--|
|2059|2059|vnc|by default not running|
|2088|2088|jupyter|by default not running|


### Connect to a container
Do not connect with the `docker exec` cmd directly to the container. Some environment variables needs to be set! Instead,
use the following proper command.
```bash
$ ./bin/container-connect [<docker exec args>]
```

### Stop a container
Therefore, the container runs detached, you need to stop the container session.
The container is reset with the exception of the mounted volumes.
```bash
$ ./bin/container-stop
```

## Services and Commands
Starting a container with `./container-start` makes all argument-scripts, located in `bin/arguments` avalibale as generell commands inside the container. Make sure you applied executable permissions to all scripts in `bin/` and its subfolders with `chmod -R +x bin/`. 

### Start GUI Service with x11vnc
By default, the vnc and gui service isn't running. Connect your terminal and run the following command. Runs the `x11vnc` service in foreground.
```bash
$ start-vnc [<x11vnc args>]
```
### Start jupyter Server
By default, jupyter isn't running. Connect your terminal and run the following command. Runs the `jupyter notebook` service in foreground.
```bash
$ start-jupyter [<juypter notebook args>]
```
**Note:** Jupyter web-service root directory is `/mnt`. However, all changes will be lost outside of the given volume folders.

## Work with jupyter notebooks trough a SSH tunnel
If the container instance runs on one of the university's deep gear machines, port forwarding via `ssh` allows to edit the notebooks on a local machine.
```bash
$ ssh -L 2088:<remote ip>:2088 <username>@<remote ip>
```

## Python modules and jupyter notebooks
### Project modules and namespace path
Some core functionalities are moved outside the jupypter notebooks. Modules, classes and as well functions located in `src/utils` are available under the python namespace name `utils`. 

**Note:** `utils` is just a namespace. If you wanna use a module from the `utils` namespace, you need to import each module separately like the following code example. The `*`-operator is not defined!
```python
# Import module specified in './src/utils' 
from utils import xyz
```

## Maintained and Developed by
- David Schwenke @ schwenkedavid@t-online.de
- Marvin Fette @ fettemarvin@gmail.com
