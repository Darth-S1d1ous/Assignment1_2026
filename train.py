from TrainTools.train import train

PROJECT_ROOT = "D:/hku/COMP 4329/Assignment1_2026"

results = train(
    # ── data paths (must match preprocess outputs) ──────────────────────
    train_npz       = "_data/train.npz",
    dev_npz         = "_data/dev.npz",
    word_emb_json   = "_data/word_emb.json",
    char_emb_json   = "_data/char_emb.json",
    train_eval_json = "_data/train_eval.json",
    dev_eval_json   = "_data/dev_eval.json",
    save_dir        = "_model",
    log_dir         = "_log",

    # ── training loop ────────────────────────────────────────────────────
    num_steps  = 60000,
    batch_size = 8,
    seed       = 42,

    # ── vanilla recipe: SGD, no scheduler, NLL loss ───────────────────────
    optimizer_name = "sgd",
    scheduler_name = "none",
    loss_name      = "qa_nll",
)

print(f"Best F1: {results['best_f1']:.4f}  |  Best EM: {results['best_em']:.4f}")