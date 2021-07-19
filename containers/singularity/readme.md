# Plato Singularity 
Singularity images offer a convenient way to quickly get started using Plato.  
## Prerequisites:
1. [Singularity](https://sylabs.io/guides/3.8/user-guide/quick_start) version 3.8 must be installed and running.
2. [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is required if users wish to access GPU hardware in the Singularity container.  Nvidia Docker is only available in Linux and Windows (using [Windows Subsystem Linux 2](https://docs.microsoft.com/en-us/windows/wsl/install-win10)).

## Usage:
A typical workflow is:
1. Create the problem definition.  Users can clone/download the [Plato Engine](https://github.com/platoengine/platoengine/tree/docker) repository which has a collection of example problems.  More experienced users may be interested in the [Use Cases](https://github.com/platoengine/use_cases) repository.
2. Start the Plato container following the instructions below and run the optimization problem(s) of interest.  The container mounts the problem directory so results are available in the host filesystem.
3. Exit the container (or just open another terminal) to visualize the results. 

### Starting a Plato container
Images are available at [cloud.sylabs.io](https://cloud.sylabs.io) for the 'release' and 'develop' branches:  
- jrobbin/default/plato-analyze:cpu-develop
- jrobbin/default/plato-analyze:cpu-release
- jrobbin/default/plato-analyze:cuda-10.2-cc-7.5-develop
- jrobbin/default/plato-analyze:cuda-10.2-cc-7.5-release

**Run the image:** From the directory that contains the problem(s) to be run, start the desired singularity container:
```shell
singularity run --nv library://jrobbin/default/plato-analyze:cuda-10.2-cc-7.5-develop
```
**Run the problem(s):**
```shell
plato-cli optimize --input plato_input_deck.i
```

Users familiar with Docker will note a few differences:
1. The singularity container starts in the host file system rather than a virtual space inside the container.  Any additions, deletions, or changes happen in the host volume.
2. Your user is the same as on the host.  That is, importantly, you're not root.  Files that you create while in the container belong to your user.
