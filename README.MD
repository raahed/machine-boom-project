# Machine Boom Lowest Point Prediction
## Project Documentation
Beside this README file the project documentation is stored in Overleaf --> https://www.overleaf.com/read/ftppxzzkjsmf

## Docker Interface
### Volume folders
Inside the docker container there are several mounted directories:
- `/mnt/simulation` mounted from `./src/simulation`: Contains the hose simulation
- `/mnt/data` mounted from `./data`: Destination of all simulation exports

### Build a Image
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
|5900|2059|vnc|by default not running|
|8888|2088|jupyter|by default not running|


### Connect to a container
Do not connect with the `docker exec` cmd directly to the container. Some environment variables needs to be set! Instead,
use the proper
```bash
$ ./bin/container-connect [<docker exec args>]
```
command.

#### Start GUI Service with x11vnc
By default, the vnc and gui service isn't running. Log into the container and run the program in the background if needed.
`-create` Flag is required to create a virtual display.

```bash
$ x11vnc -create -once &
```

### Stop a container
Therefore, that the container runs detached, you need to stop the container session.
The container will be rested excepted the mounted volumes.
```bash
$ ./bin/container-stop
```

## Maintained and Developed by
- David Schwenke @ schwenkedavid@t-online.de
- Marvin Fette @ fettemarvin@gmail.com
