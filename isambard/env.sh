# source isambard/env.sh   (login node and inside jobs)
export PROJ=$PROJECTDIR/$USER                 # per-user area in the project space (models, data, images)
export HF_HOME=$PROJ/hf                       # HF cache; populated on the login node, read-only in jobs
export REPO=${REPO:-$HOME/PLC-DPO}            # this clone (code + data/ + configs/)
export SIF=$PROJ/sif/plcdpo-train.sif
export SIF_EVAL=$PROJ/sif/plcdpo-eval.sif
export OUT=$SCRATCHDIR/plcdpo                 # adapters, generations, scores
export PLCDPO_NO_UNSLOTH=1
export APPTAINER_CACHEDIR=$SCRATCHDIR/apptainer/cache APPTAINER_TMPDIR=$SCRATCHDIR/apptainer/tmp   # keep image layers/tmp off the 100 GB $HOME
mkdir -p "$HF_HOME" "$PROJ/sif" "$OUT" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
# run a command inside the training image with GPU, binding the three storage areas
run_in_sif() { apptainer exec --nv -B "$PROJECTDIR","$SCRATCHDIR","$HOME" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}",PLCDPO_NO_UNSLOTH=1 "$SIF" "$@"; }
