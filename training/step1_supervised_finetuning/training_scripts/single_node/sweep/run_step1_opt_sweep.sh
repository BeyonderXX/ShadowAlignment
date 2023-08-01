#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
for z in {2..3}
do
    for offload in true false
    do
        for lora in true false
        do
            cmd="bash training_scripts/single_node/sweep/run_1.3b_lora_swp.sh \
                ${z} \
                ${offload} \
                ${lora} \
                step1_z${z}_offload_${offload}_lora_${lora}"
            echo "----------------------------- CALLING SHELL SCRIPT -----------------------------"
            echo $cmd
            $cmd
            pkill -9 python
            sleep 60
            echo ""
        done
    done
done
