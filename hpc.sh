#!/bin/sh
# Moab-PBS staging file
#PBS -N rnn
#PBS -q rudens
#PBS -l nodes=1:ppn=8,feature=centos7,walltime=96:00:00
#PBS -j oe
##PBS -t 1-10 nodes=wn57:ppn=8:gpus=1,feature=centos7,walltime=96:00:00
## -F "relu tf"


module load cuda/cuda-7.5
export TMPDIR=$HOME/tmp
source $HOME/machinelearn/bin/activate
cd $HOME/rnn/
python main.py $1




