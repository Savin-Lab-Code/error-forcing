################################################################################
# run_one.sh  
# Usage: run_one.sh --B 0 ... --seed 154 --task MDXOR
################################################################################

eval "$($HOME/miniforge3/bin/conda shell.bash hook)" \
    && conda activate EFneurips
export CUDA_VISIBLE_DEVICES=""

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
set -x

base=$(printf '%s\n' "$*" \
       | tr -c 'A-Za-z0-9_.=-' '_' )     

LOGDIR="$(dirname "$0")/logs"
mkdir -p "$LOGDIR"

OUTFILE="$LOGDIR/${base}.out"
ERRFILE="$LOGDIR/${base}.err"

exec > >(tee  -a "$OUTFILE") \
     2> >(tee  -a "$ERRFILE" >&2)

echo "[$(date +'%F %T')] Starting configuration:"
echo "  $*"
echo "  stdout → $OUTFILE"
echo "  stderr → $ERRFILE"
echo "-------------------------------------------------------------"

# Run the training job 
python -u "$(dirname "$0")/train_single.py" "$@"
STATUS=$?

echo "-------------------------------------------------------------"
echo "[$(date +'%F %T')] Finished with exit‑code $STATUS"
echo

# Append OK / FAIL to sweep_results.txt 
RESULT_FILE="$(dirname "$0")/sweep_results.txt"
if [ $STATUS -eq 0 ]; then
    printf "OK  %s\n" "$*" >> "$RESULT_FILE"
else
    printf "FAIL  %s\n" "$*" >> "$RESULT_FILE"
fi

exit $STATUS
