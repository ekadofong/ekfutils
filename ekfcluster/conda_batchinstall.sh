conda init zsh

ENVIRONMENTS="$1"
PACKAGE="$2"

for KEY in $ENVIRONMENTS
do
    echo installing $PACKAGE in $KEY
    conda activate $KEY && conda install $PACKAGE && conda deactivate
done