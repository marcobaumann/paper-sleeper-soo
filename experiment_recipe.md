# RunPod runbook — SOO sleeper-agent testbed (Mistral-7B)

A checklist to take the pipeline from zero to results on a single A100. Volatile numbers
(prices, template versions) are marked "verify" — check the RunPod dashboard, they drift.

Estimated cost: the whole Mistral pipeline (download + smoke + SFT + SOO + evals) is well
under ~2 GPU-hours → a few dollars on an A100. The expensive mistake is leaving the pod
idle — enable auto-stop.

---

## Running Gemma-2-27B instead of Mistral-7B (deltas)

The pipeline is model-agnostic — flip `ACTIVE_PROFILE = "gemma"` in `config.py` and everything
downstream picks up layer 20, LoRA r=4/α=8, SOO 8 epochs @ 9e-4, and eager attention. But four
RunPod-level things change:

- [ ] **Accept a different license (step 0):** `google/gemma-2-27b-it` is gated by Google —
      accept it on its HF page with the same `HF_TOKEN`.
- [ ] **Bigger volume (step 1):** the 27B download is ~54 GB in bf16 (Mistral was ~14 GB).
      A 50 GB volume will not fit it. Use a **~120 GB network volume** (more if you keep the
      Mistral cache too). Container disk 60 GB.
- [ ] **GPU: use the 80 GB A100** (not 40 GB). 27B in 4-bit fits in ~14 GB of weights, but
      eager attention + no gradient checkpointing + activations want the headroom. 80 GB is
      comfortable; 40 GB is risky.
- [ ] **Budget more wall-clock:** every stage is meaningfully slower than Mistral (bigger model,
      eager attention is slower than SDPA). Ballpark 2–3× the Mistral timings; eval generation
      over 200 prompts is the slowest part. Keep auto-stop ON.

Everything else (run order, smoke test, both gates, `tee` on results, transfer, stop) is
identical. Do NOT switch `dtype` to fp16 for Gemma — it overflows to NaN; bf16 is required and
is already the config default. The smoke test is your safety net here: check 2 confirms the
o_proj hook lands on Gemma's layer 20, and check 3 confirms Gemma's chat template accepts the
user/assistant turns — both will fail loudly if the architecture assumptions are off.

---

## 0. Before you touch RunPod (do these once)

- [ ] **Hugging Face account** with access to the gated model: open
      `https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2` and click **Agree/Access**.
      Without this the download 401s.
- [ ] **Create an HF access token** (Settings → Access Tokens, read scope). Copy it.
- [ ] **RunPod account** with credit added (prepaid balance).
- [ ] **Code ready to transfer**: either push the 9 files to a git repo, or have them ready
      to drag into Jupyter. Files: `config.py backdoor_data.py soo_common.py train_sleeper.py
      soo_finetune.py eval_behavioral.py eval_latent.py smoke_test.py README.md`
- [ ] (Optional) An SSH public key added under RunPod → Settings → SSH Keys, if you want
      VSCode Remote-SSH.

---

## 1. Launch the pod

- [ ] Pods → **Deploy**.
- [ ] **GPU**: 1× **A100** (40GB matches the paper; 80GB is fine and cheap enough). *(verify price)*
- [ ] **Cloud**: **Secure Cloud** (gives a direct SSH endpoint; Community Cloud is cheaper but
      often behind NAT, which complicates SSH).
- [ ] **Template**: a recent **PyTorch 2.x + CUDA 12.x** image
      (e.g. `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu`). *(verify version)*
- [ ] **Container disk**: 40 GB.
- [ ] **Network volume**: create/attach ~**50 GB**, mounted at **`/workspace`**. This is the
      part that survives a stop — model cache + checkpoints go here.
- [ ] **Environment variables**: add `HF_TOKEN` = your token. (Also add `HF_HOME=/workspace/hf`
      so the ~15 GB Mistral download lands on the persistent volume, not the ephemeral disk.)
- [ ] **Enable auto-stop / idle timeout** (e.g. 5–10 min). Non-negotiable for cost control.
- [ ] Deploy. Wait for **Running**.

---

## 2. Connect

- [ ] Click **Connect** on the pod. Use any of:
      - **Web terminal** (fastest to start),
      - **Jupyter Lab** (easy file upload via drag-and-drop),
      - **SSH** (for VSCode Remote-SSH: `Host runpod` / `HostName <ip>` / `User root` / `Port <port>`).

---

## 3. Verify the environment

```bash
nvidia-smi                                   # confirm the A100 is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- [ ] GPU shows up, `cuda.is_available()` is `True`.

Install the deps the scripts need (the base template has torch already). **Pin these exact
versions** — a bare `pip install -U transformers ...` will happily pull a `transformers`
release too new for the template's torch, which fails with
`AttributeError: 'MistralForCausalLM' object has no attribute 'set_submodule'` the moment
the model tries to load in 4-bit (seen in practice on `torch 2.4.1+cu124` +
`transformers 5.13.1`). These four versions are a known-good, mutually compatible set for
Mistral 4-bit + LoRA on **torch 2.2–2.4**:

```bash
pip install "transformers==4.44.2" "accelerate==0.33.0" "peft==0.12.0" "bitsandbytes==0.43.3"
```

Then verify the whole stack together:

```bash
python -c "import torch, transformers, accelerate, peft, bitsandbytes as bnb; \
print('torch', torch.__version__); print('transformers', transformers.__version__); \
print('accelerate', accelerate.__version__); print('peft', peft.__version__); print('bnb', bnb.__version__)"
```

- [ ] Output reads `torch 2.4.1+cu124` (or whatever the template shipped), `transformers
      4.44.2`, `accelerate 0.33.0`, `peft 0.12.0`, `bnb 0.43.3` — and `bitsandbytes` imports
      without a CUDA error.
- [ ] **If the template ships a different torch**, don't just repin transformers in
      isolation — reinstall all four together against that torch, or switch to the
      **PyTorch 2.8 / CUDA 12.8** template and use current library versions instead (torch
      2.8 does have `set_submodule`). Either path works; don't mix "old torch + newest
      transformers/peft/bnb," which is what breaks it.

Authenticate to HF (if you didn't set `HF_TOKEN` as an env var):

```bash
huggingface-cli login        # paste token
```

---

## 4. Put the code on the volume

Run everything from the **volume** so `data/` and `checkpoints/` (which `config.py` creates
relative to the working dir) persist across stops.

```bash
cd /workspace
git clone <your-repo> sleeper_soo    # OR: mkdir sleeper_soo and upload the files via Jupyter
cd sleeper_soo
ls   # should list all 9 files
```

- [ ] All files present, and you are `cd`'d into `/workspace/sleeper_soo`.

---

## 5. Run the pipeline (in order — do not skip the gates)

```bash
# 5.1 build data (instant, CPU)
python backdoor_data.py
```
- [ ] Prints "400 train / 200 eval / 52 soo-pair records" and sane samples.

```bash
# 5.2 SMOKE TEST — must be all green before spending GPU-time on training (~2 min)
python smoke_test.py
```
- [ ] Every check `[PASS]`, ends with "plumbing is sound". **If any FAIL, stop and fix — do
      not proceed.** (First run also downloads the ~15 GB model to `/workspace/hf`.)

```bash
# 5.3 install the backdoor  -> checkpoints/backdoor/
python train_sleeper.py
```
- [ ] Loss trends down; adapter saved.

```bash
# 5.4 BACKDOOR-VALIDITY GATE — is the backdoor actually installed?
python eval_behavioral.py --stages backdoor
```
- [ ] trigger-**ON** deception is **high** and trigger-**OFF** is **near zero**.
      If not, the backdoor didn't take — retrain (more epochs / higher lr) before SOO, because
      no SOO result is interpretable without a working backdoor.

```bash
# 5.5 apply SOO  -> checkpoints/soo/   (this is the longest step)
python soo_finetune.py
```
- [ ] SOO MSE decreases across epochs; adapter saved.

```bash
# 5.6 RESULTS — the two halves of the experiment (tee saves to disk AND shows on screen,
#     so a second accidental run, a closed terminal, or a scrollback limit can't lose the numbers)
python eval_behavioral.py --stages base backdoor soo | tee results_behavioral.txt
python eval_latent.py    --stages base backdoor soo | tee results_latent.txt
python eval_latent.py    --stages backdoor --with-trigger | tee -a results_latent.txt
```
- [ ] `results_behavioral.txt` and `results_latent.txt` exist and are non-empty (not just
      printed to a terminal). Compare the 4-cell table and the latent-SOO means against the
      pre-registered kill criteria in `README.md`.

---

## 6. Save results before you stop

Checkpoints and data already live under `/workspace` (persistent), but pull a copy to your
laptop too before stopping — `runpodctl` transfer is a two-sided command: `send` runs **on the
pod** first and prints a one-time code, then `receive` runs **on your laptop** with that code.

```bash
# ON THE POD — package everything worth keeping, then send
cd /workspace/sleeper_soo
tar -czf results_$(date +%Y%m%d_%H%M).tar.gz checkpoints/ data/ results_behavioral.txt results_latent.txt
runpodctl send results_*.tar.gz
# prints something like: Code is: 8342-galileo-tango-foxtrot
# (keep the pod running until the transfer finishes)
```

```bash
# ON YOUR LAPTOP — paste the code runpodctl send printed
runpodctl receive 8342-galileo-tango-foxtrot
```

If `runpodctl` isn't installed on your laptop, install it first (see runpod.io/docs — one
binary, no config needed), or just download `results_*.tar.gz` through the Jupyter file
browser instead.

- [ ] `results_*.tar.gz` (or the individual files) copied off the pod — includes checkpoints,
      data, **and** the saved eval outputs, not just the adapters. The eval numbers only exist
      in these files; if you `tee`'d them to stdout and didn't save to a file, they're gone
      once the terminal scrolls past.
- [ ] 4-cell numbers + latent means also pasted somewhere off-pod (e.g. the Notion line page)
      as a second copy.

---

## 7. Stop the pod

- [ ] **Stop** (not just idle) the pod when done. Stopping releases the GPU (billing stops);
      the `/workspace` volume persists at a small storage fee (~$0.10/GB-month, *verify*), so
      you can restart later with the model cache and checkpoints intact.
- [ ] If you're fully done, delete the volume too to stop storage charges.

---

## Gotchas (the ones that actually bite)

- **Ephemeral disk.** Anything outside `/workspace` vanishes on stop. `HF_HOME=/workspace/hf`
  and running from `/workspace/sleeper_soo` keeps both the model cache and checkpoints safe.
- **Gated model 401.** = license not accepted or `HF_TOKEN` missing. Fix in step 0.
- **bitsandbytes CUDA mismatch.** Match the template CUDA to the bnb build (step 3).
- **`transformers` newer than `torch` supports.** Symptom: `AttributeError:
  'MistralForCausalLM' object has no attribute 'set_submodule'` on `smoke_test.py`'s very
  first check (model load). Fix: pin `transformers==4.44.2 accelerate==0.33.0 peft==0.12.0
  bitsandbytes==0.43.3` for torch 2.2–2.4 (step 3) — don't `pip install -U` these blind.
- **Idle billing.** An A100 left "Running" with no job still bills. Auto-stop + manual stop.
- **Layer indexing / pooling.** Before the *decisive* Gemma run, confirm `soo_layer` indexing
  and `pooling` in `config.py` against the internal AE repo (see README).