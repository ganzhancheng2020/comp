# AI competition survey and strategy (as of 2026-09-26)

Goal: find AI competitions this project can realistically enter, build the entries, and win prize money.

## Constraints discovered in this environment

* **Network:** the cloud container can only reach GitHub and package registries (PyPI, npm).
  Kaggle, HuggingFace, Zindi, DrivenData, Tianchi, iFLYTEK, Devpost, arcprize.org and
  mlcontests.com are all **blocked** by the environment's network policy. So data for most
  competitions can't be downloaded, and nothing can be submitted from here.
* **Compute:** 4 CPU cores, 15 GB RAM, no GPU.
* **Identity:** every prize needs a registered human participant (a Kaggle or Devpost account,
  age/eligibility checks, tax forms). **Registration, submission and payout have to be done by
  you.** The code here is built so that each of those steps is a short manual action.

## Competitions surveyed

| Competition | Prize | Deadline | Feasible here? | Verdict |
|---|---|---|---|---|
| **ARC Prize 2026 – Paper Track** (Kaggle) | $75k guaranteed pool, plus a $375k bonus pool for papers scoring >4.5/5 | 2026-11-02 (entry deadline 10-26) | **Yes.** Data is on GitHub; the linked code submission "need not achieve a high score" | **Primary target** |
| ARC Prize 2026 – ARC-AGI-2 (Kaggle) | $700k pool, $75k top progress prize | 2026-11-02 (entry 10-26) | Partly. Our CPU solver runs, but top-5 needs roughly 25%+ | Required as the Paper Track's linked code submission |
| ARC Prize 2026 – ARC-AGI-3 | $850k; Milestone #2 on 09-30 | 11-02 | No. The game API host is blocked, it needs a GPU, and the milestone is 4 days away | Skip |
| OpenCV AI Competition 2026 | $5k / $3k / $2k plus 2 × $1k | 10-26 | No. The mandatory proposal closed 08-17 | Closed |
| Agents for Humans Hackathon (AWS / Devpost) | $40k | 09-14 | No. Deadline passed | Closed |
| AI Cup 2026 (Kaggle, NL) | €7k | — | No. Students/recent graduates of NL programs only | Ineligible |
| CCF BDCI 2026 | ¥20k / ¥10k / ¥5k per topic | Prelims Aug–Oct | Unlikely. Finals are in October | Probably closed |
| iFLYTEK AI Developer Contest 2026 | Varies per topic (often ¥10k–¥50k) | Rolling | Only if you download the data and add it to the repo, or widen network access | **Best secondary option** |
| **IEEE Big Data Cup 2026 – TrafficFlowBench** (Kaggle) | $1,500 / $1,000 / $500 + $500 student, per division | 2026-11-06 | Yes, **if Kaggle access is enabled**. Tabular + traffic physics, CPU-friendly, about 179 teams, open-source solutions at 0.845–0.867 public LB to build on | **Best secondary target** |
| IEEE Big Data Cup 2026 – AI Emulation (CarbonGlobe, Kaggle) | $1,500 / $1,000 / $500 + $500 student, per division | 2026-11-06 | Only with Kaggle access; large dataset; about 65 teams | Tertiary |
| IEEE Big Data Cup 2026 – FinReason Cup | **No cash prizes** (certificates only) | 10-15 | Data is on GitHub | Skip: no money |
| Devpost AI hackathons (various) | Often gift cards or small cash | Rolling | Possible, but they need a demo video and your account | Opportunistic |

## Chosen plan

1. **ARC Prize 2026 Paper Track plus the linked ARC-AGI-2 code submission.**
   * `arcsolver/` is a CPU-only solver. It combines grid-transform search with
     *MDL local-rule induction* and Bayesian model averaging.
   * `analysis_oracle.py` is the paper's main empirical contribution. It separates
     *expressibility* (a compact rule exists, found with an oracle that sees the test outputs)
     from *learnability* (the solver finds it from the training pairs alone).
   * `kaggle/arc_mdl_rules.ipynb` is the self-contained offline notebook for the Kaggle
     code submission.
   * `paper/` holds the write-up for the Paper Track.
2. **Secondary: TrafficFlowBench** (IEEE Big Data Cup 2026, deadline Nov 6). The official
   toolkit (`github.com/jacky850/trafficflowbench-public`) and strong open-source entries are
   reachable, but the data is only on Kaggle. With Kaggle access plus an API token, the whole
   loop runs from a session: download, train, validate on the train months, and submit with
   `kaggle competitions submit`.
3. **Tertiary:** iFLYTEK algorithm topics (requires `challenge.xfyun.cn` access or manually
   committed data).

## What you need to do (the steps that cannot be automated from here)

1. Create or sign in to a Kaggle account (18+). Join **ARC Prize 2026 – ARC-AGI-2** and
   **ARC Prize 2026 – Paper Track**, and accept the rules **before 2026-10-26**.
2. Upload `kaggle/arc_mdl_rules.ipynb` as a Kaggle notebook. Attach the competition data,
   turn internet off, click *Submit*. Note the submission ID.
3. Submit the paper from `paper/` on the Paper Track, linked to that code submission, by
   **2026-11-02**. Make this GitHub repository public, because prize eligibility requires open source.
4. **Unlock TrafficFlowBench (highest-value step):** join
   `kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench` and accept its rules. Then, in
   the cloud environment settings (environment menu in the session title bar → Edit):
   * add `kaggle.com`, `www.kaggle.com` and `storage.googleapis.com` to the allowed network domains;
   * add environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`, from Kaggle → Settings →
     API → Create New Token. Do **not** paste the token into chat.

   A new session then picks these up and can build and submit the entry end to end.

## Honest odds

The ARC-AGI-2 leaderboard prizes need about 25%+ accuracy on the private set. That takes GPU
test-time training of LLMs, which this environment can't do. The Paper Track is scored on
six criteria (accuracy, universality, progress, theory, completeness, novelty), so a paper
with a clear analytical contribution has a real, if modest, chance at a runner-up award.
Expect low single-digit odds for any single entry. Entering more competitions, which needs the
network access above, is the main way to improve the expected payout.
