# Machine Boom Lowest Point Prediction
## Project Documentation
Beside this README file the project documentation is stored in Overleaf --> https://www.overleaf.com/read/ftppxzzkjsmf

## Docker Interface
### Build a Image
The bash script runs based on the `.env` arguements. However, the script passes thoses arguments to `docker buildx`. Image tag will be `machine-boom-project` after all.

```bash
$ ./bin/build-image
```

#### Requirments before build a image
- Place a license file in the project root folder. Note: Make sure the file the the correct extension. Each `.lic` file will be copied into the container
- Specify agx version and distirbution in `.env`. Versions listed at [Algoryx website](https://www.algoryx.se/documentation/complete/agx/tags/latest/doc/UserManual/source/changelog.html)

### Run a container
continued

### Stop a container
continued

## Start GUI Service with x11vnc
By default the vnc and gui service isn't running. Log in with ssh credentials and run the program in the background if needed.
`-create` Flag is requiered to create a virtual display.
```bash
$ x11vnc -create -once &
```

## Maintained and Developed by
- David Schwenke @ schwenkedavid@t-online.de
- Marvin Fette @ fettemarvin@gmail.com
