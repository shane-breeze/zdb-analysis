# Z invisible DB analysis

Setup with

```
source setup.sh
```

To run the analysis use `analyse.py`:

```
usage: analyse.py [-h] [-m MODE] [-j NCORES] config

positional arguments:
  config                Path to yaml file

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Parallelisation: 'multiprocessing', 'sge', 'htcondor'
  -j NCORES, --ncores NCORES
                        Number of cores for 'multiprocessing' jobs
```
