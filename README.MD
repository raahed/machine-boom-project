# Machine Boom Lowest Point Prediction
## Project Documentation
Beside this README file the project documentation is stored in Overleaf --> https://www.overleaf.com/read/ftppxzzkjsmf

## Docker Interface
### Build a Image
The bash script runs based on the `.env` arguements.
```bash
$ ./bin/build-image
```
However, the script passes thoses arguments to `docker buildx`. Image tag will be `machine-boom-project` after all.

#### Requirments before build a image
- Place a license file in the project root folder. Note: Make sure each license file has the correct extension. Each `.lic` file will be copied into the container
- Specify agx version and distirbution in `.env`. All Versions are listed at Algoryxs [Changelog](https://www.algoryx.se/documentation/complete/agx/tags/latest/doc/UserManual/source/changelog.html) on the website.

### Run a container
continued
```bash
$ ./bin/container-start
```

continued: Run Container on deep learning, pass additional docker flags
```bash
$ ./bin/container-start --gpus all
```

#### Exposed ports
A running container instance offers a number of services through exposed ports:

|Inner Port|Exposed Port|Service|Comment|
|--|--|--|--|
|22|2022|ssh||
|5900|2059|vnc|by default not running|
|8888|2088|jupyter|by default not running|

### Stop a container
continued
```bash
$ ./bin/container-stop
```

## Start GUI Service with x11vnc
By default the vnc and gui service isn't running. Log in with ssh credentials and run the program in the background if needed.
`-create` Flag is requiered to create a virtual display.
```bash
$ x11vnc -create -once &
```

## Maintained and Developed by
- David Schwenke @ schwenkedavid@t-online.de
- Marvin Fette @ fettemarvin@gmail.com
