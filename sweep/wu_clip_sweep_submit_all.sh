#!/usr/bin/env bash

set -euo pipefail

# fill in sweep IDs from yaml output
export SWEEP_ID_H100="adinas-wcm/ssl_project-sweep/8xjw7z9t"
export SWEEP_ID_L40="adinas-wcm/ssl_project-sweep/b2du15jr"

jid_h100=$(sbatch --export=ALL,SWEEP_ID_H100 --parsable /home/ads4015/ssl_project/sweep/wu_clip_sweep_agent_2gpu_h100.sh || true)
echo "Submitted H100 agent: ${jid_h100:-pending or failed}"

jid_l40a=$(sbatch --export=ALL,SWEEP_ID_L40 --parsable /home/ads4015/ssl_project/sweep/wu_clip_sweep_agent_2gpu_l40.sh)
jid_l40b=$(sbatch --export=ALL,SWEEP_ID_L40 --parsable /home/ads4015/ssl_project/sweep/wu_clip_sweep_agent_2gpu_l40.sh)
echo "Submitted L40 agents: $jid_l40a, $jid_l40b"

echo "Monitor: squeue -u $USER -o '%i %9P %9j %2t %R %8b'"












