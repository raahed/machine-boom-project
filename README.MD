# Machine Boom Lowest Point Prediction
## Project spaces
- Git repository:  [Gitlab NT](https://gitsvn-nt.oru.se/mnfe/machine-boom-project)
- Project report: [Overleaf project](https://www.overleaf.com/read/ftppxzzkjsmf)
- Data drive: [Oru Cloud](https://cloud.oru.se/s/sC4BxzntTFNqxZB)

## Docker Interface
### Mounted volume folders
Inside the docker container there are several mounted directories:
|volume|mounted folder|comment|
|--|--|--|
| `/mnt/src/` | `src/` | Contains main sources |
| `/mnt/scripts/` | `scripts/` | Python scripts
| `/mnt/models` | `models/` | Saved models storage |
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

#### Published ports
A running container instance offers a number of services through exposed ports:

|Inner Port|Outer Port|Service|Comment|
|--|--|--|--|
|2059|2059|vnc|by default not operating|
|2088|2088|jupyter|by default not operating|
|-|2092-2099|ray|only exposed|
|-|10000-20000|ray|only exposed|

**Note:** exposed ports only visible to the host or other docker containers.

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

### Start GUI service provided by x11vnc
By default, the vnc and gui service isn't running. Connect your terminal and run the following command. Runs the `x11vnc` service in foreground.
```bash
$ start-vnc [<x11vnc args>]
```
### Start a jupyter Server
By default, jupyter isn't running. Connect your terminal and run the following command. Runs the `jupyter notebook` service in foreground.
```bash
$ start-jupyter [<juypter notebook args>]
```
**Note:** Jupyter web-service root directory is `/mnt/src/notebooks`. However, all changes will be lost outside of the given volume folders.

## Work with jupyter notebooks through a SSH tunnel
If the container instance runs on one of the university's deep gear machines, port forwarding via `ssh` allows to edit the notebooks on a local machine.
```bash
$ ssh -L 2088:<remote ip>:2088 <username>@<remote ip>
```

## Python modules and jupyter notebooks
### Project modules and namespace path
Some core functionalities are moved outside the jupypter notebooks. Modules, classes and as well functions located in `src/utils` or `src/models` are available under the python namespace name `utils` or `models`. 

**Note:** `utils` or `models` is just a namespace. If you wanna use a module from the `utils` or `models` namespace, you need to import each module separately like the following code example. The `*`-operator is not defined!

Include the following import statement above your code:
```python
# Import local modules from 'src/*' as package 'utils' or 'models'
import sys; sys.path.insert(0, '../')
```

Example: Import a util with
```python
# Import a method from 'src/utils/xyz.py' 
from utils.xyz import method
```

## Traning with multiple machines using Docker swarm and ray
The goal is to connect multiple docker containers running on different machines and running *ray engine* with a cluster manager node.

### Prerequisites
- Make sure the project image is build on all machines, that should participate in the cluster.
- To connect the machines properly, read the docker swarm tutorial regarding port policy or firewall settings ([Docker documentation](https://docs.docker.com/engine/swarm/swarm-tutorial/#open-protocols-and-ports-between-the-hosts)).

### 1. Step: Init Docker swarm network
Choose a machine as swarm manager and initialize the network. All swarm nodes must reach the advertise address of the manager node.

```bash
$ docker swarm init --advertise-addr <reachable ip>
```

Using the given join token to join the swarm network.
```bash
$ docker swarm join --token <token> <reachable ip>:2377
```

For more, follow the steps in the [Docker swarm tutorial](https://docs.docker.com/engine/swarm/swarm-tutorial/create-swarm/).

### 2. Step: Create an overlay network
Therefore the setup not using services among the nodes we need to set up a network manually.

Instance a new network on the manager node.
```bash
$ docker network create --driver overlay --attachable <network name>
```

### 3. Step: Start containers
Start and attach the container to the swarm network. Passing the `--gpus all` flag for cuda support. Make this step on each machine in the swarm network.
```bash
$ ./bin/container-start --gpus all --network <network name>
```

### 4. Step: Instance the ray engine
Connect to the containers and start the ray engine. Make sure the head is matching the environment variable `CLUSTER_HEAD_NAME` given in `.env`. All worker nodes will try to connect to this container name.

The script will decide by the `hostname` to start or trying to join a cluster. Bypass this behavior by using the optional parameter `--head` to create a head node anyway.

**Note** Make sure you start the engine firstly on the cluster head node.
```bash
$ cluster-join [--head]
```

The all connected ray nodes by using the `ray status` command or by calling the `utils.get_cluster_nodes()` method.

Make sure to terminate the ray engine properly after using it.
```bash
$ cluster-leave
```

**Referenz** [Ray CLI Documentation](https://docs.ray.io/en/latest/cluster/cli.html)

## Maintained and Developed by
- David Schwenke @ schwenkedavid@t-online.de
- Marvin Fette @ fettemarvin@gmail.com
