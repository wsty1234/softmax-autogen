# softmax-autogen

This is an experiment to use the LLM do triton kernel gen.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `softmax-autogen/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b softmax-autogen/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `run_test_softmax.py` — script to eval the result of triton same with the torch. Do not modify.
   - `calc_softmax_speedup.py` - evaluates the speed of the triton kernel. Do not modify.
   - `fused_softmax.py` — the file you modify. the implemettion of the trition kenrel.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

The goal is to find the best softmax triton kernel.

**What you CAN do:**
- Modify `fused_softmax.py` — this is the only file you edit. Attention: in this file you do not modify the test_softmax and benchmark_softmax functions. you can modify the softmax and _softmax_kernel and add some new functions. 

**What you CANNOT do:**
- Modify `run_test_softmax.py`. It is read-only. 
- Modify `calc_softmax_speedup.py`. It is read-only. 

**The goal is to find the best softmax triton kernel.** make the triton kernel faster and faster. But always keep in mind that the result of the triton kernel should be the same as the torch.


## Output format

there are multiple outputs:

Once checked the equality of the torch and triton kernel, it will print the following:

```
passed
passed
passed
passed
passed
passed
passed
passed
passed
passed
```

Note if the output have any word except "passed", it means the test failed. you need to start a new experiment loop.

The other output will be:

```
triton average: 820.080991
torch average: 838.461689
triton speedup over torch: -2.19%
```

Note the number of triton speedup over torch is the percentage of speedup over torch. positive number means the triton kernel is faster than the torch. bigger is better.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `softmax-autogen/mar23`).

always activate the venv `conda activate unik3d` before you run the python scripts.

LOOP:

1. Look at the git state: the current branch/commit we're on
2. Tune `fused_softmax.py` with an experimental idea by directly hacking the code.
3. `git add ./fused_softmax.py` (never git add -A or git add .)
4. find if the `softmax-kernel.csv` is existed, if existed, `rm softmax-kernel.csv`
5. Run `python run_test_softmax.py` (you can see the first in output format) check the output all passed. if not, go back to step 2.
6. Run `python fused_softmax.py`
7. check if the file `softmax-kernel.csv` is existed, if not then wait. if yes, continue.
8. Run `python calc_softmax_speedup.py` (you can see the second in output format) check the output if the number of triton speedup over torch is positive or bigger than -0.5% then continue. else go back to step 2.
9. git commit -m "`the number of triton speedup over torch`" 
10. git push
11. end the loop


The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.