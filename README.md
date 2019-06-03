# Z invisible DB analysis

Setup with

```
source setup.sh
```

To run the analysis use `zdb_analyse.py`:

```
usage: zdb_analyse.py [-h] [-m MODE] [-j NCORES] [--sge-opts SGE_OPTS]
                      [-n NFILES] [-o OUTPUT]
                      config

positional arguments:
  config                Path to yaml file

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Parallelisation: 'multiprocessing', 'sge', 'htcondor'
  -j NCORES, --ncores NCORES
                        Number of cores for 'multiprocessing' jobs
  --sge-opts SGE_OPTS   Options to pass onto qsub
  -n NFILES, --nfiles NFILES
                        Number of files to process. -1 = all
  -o OUTPUT, --output OUTPUT
                        Output file

```
