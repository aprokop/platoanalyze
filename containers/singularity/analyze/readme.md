## To build:

```shell
sudo singularity build plato-analyze-cc-75.def plato-analyze-cc-75.sif
```

## To use

### Local image:

```shell
singularity run --nv plato-analyze-cc-75.sif
```

### Public image:

```shell
singularity run --nv library://jrobbin/default/plato-analyze-cc-75:latest
```

## To publish an image:

### Sign the image
```shell
singularity sign plato-analyze-cc-75.sif
```

### Push the image
```shell
singularity push -D "Plato Analyze CC 7.5" plato-analyze-cc-75.sif library://jrobbin/default/plato-analyze:cuda-cc-7.5-develop
```
