import modal
import os

# ── 1. Modal image with all dependencies ──────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow[and-cuda]",
        "keras",
        "scikit-learn",
        "numpy",
        "pydot",
        "graphviz",
    )
    .apt_install("graphviz")
)

# ── 2. Persistent volume for BOTH data and results ────────────────────────────
volume = modal.Volume.from_name("earthquake-dl-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

app = modal.App("earthquake-dl-hrgnss", image=image)


# ── 3. Training function (runs on Modal GPU) ───────────────────────────────────
@app.function(
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={VOLUME_PATH: volume},
)
def train(nst: int = 7, nt: int = 501, nc: int = 3):
    import keras
    import tensorflow as tf
    import numpy as np
    import pickle
    import logging

    from sklearn.model_selection import train_test_split

    # ── Config ────────────────────────────────────────────────────────────────
    case_nm     = f"GNSS_M{nst}S_{nt}"
    data_dir    = f"{VOLUME_PATH}/data/{case_nm}/"
    dir_out     = f"{VOLUME_PATH}/results"
    dir_log     = f"{dir_out}/out_log/"
    dir_datinf  = f"{dir_out}/data_info_1/{case_nm}/"
    dir_trmodel = f"{dir_out}/models_1/{case_nm}/"
    dir_pred    = f"{dir_out}/predictions_1/{case_nm}/"

    for d in [dir_log, dir_datinf, dir_trmodel, dir_pred]:
        os.makedirs(d, exist_ok=True)

    logging.basicConfig(
        filename=f"{dir_log}{case_nm}.log",
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    def build_model(nst, nt, nc):
        from keras.models import Sequential
        from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
        from keras.constraints import MaxNorm

        model = Sequential([
            Conv2D(12,  (1,3), activation="relu", input_shape=(nst, nt, nc)),
            MaxPooling2D((1,2)),
            Conv2D(24,  (1,3), activation="relu", padding="same"),
            Conv2D(32,  (1,3), activation="relu", padding="same"),
            MaxPooling2D((1,2)),
            Conv2D(64,  (1,3), activation="relu", padding="same"),
            Conv2D(128, (1,3), activation="relu", padding="same"),
            MaxPooling2D((1,2)),
            Conv2D(256, (1,3), activation="relu", padding="same"),
            Flatten(),
            Dense(128, activation="relu", kernel_initializer="normal", kernel_constraint=MaxNorm(3)),
            Dense(32,  activation="relu", kernel_initializer="normal", kernel_constraint=MaxNorm(3)),
            Dense(1),
        ])
        return model

    def get_lr_metric(optimizer):
        def lr(y_true, y_esti):
            return optimizer.learning_rate
        return lr

    # ── Load data from volume ─────────────────────────────────────────────────
    print(f"Loading data from {data_dir}")
    x = np.load(data_dir + "xdata.npy")
    y = np.load(data_dir + "ydata.npy")
    print(f"Dataset shape: x={x.shape}, y={y.shape}")

    ix = np.arange(len(y), dtype=int)
    x_tr1, x_test, ix_tr1, ix_test = train_test_split(x, ix, test_size=0.1,  random_state=1)
    x_train, x_val, ix_train, ix_val = train_test_split(x_tr1, ix_tr1, test_size=0.2, random_state=1)
    y_train, y_val, y_test = y[ix_train], y[ix_val], y[ix_test]

    np.save(dir_datinf + "index_datatrain.npy", ix_train)
    np.save(dir_datinf + "index_dataval.npy",   ix_val)
    np.save(dir_datinf + "index_datatest.npy",  ix_test)

    logging.info("x_train=%s  x_val=%s  x_test=%s", x_train.shape, x_val.shape, x_test.shape)

    # ── Train ─────────────────────────────────────────────────────────────────
    tf.random.set_seed(2)

    batch_size      = 128
    epochs          = 200
    steps_per_epoch = max(1, len(x_train) // batch_size)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=steps_per_epoch,
        decay_rate=0.9,
        staircase=True,
    )

    opt       = keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_metric = get_lr_metric(opt)
    model     = build_model(nst, nt, nc)
    model.compile(loss="mse", optimizer=opt, metrics=["mae", lr_metric])

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=dir_trmodel + "cp.weights.h5",
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[checkpoint_cb],
        shuffle=True,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, mae, lr = model.evaluate(x_val, y_val, verbose=2)
    logging.info("loss=%.5f  mae=%.5f  lr=%.5f", loss, mae, lr)
    np.savetxt(dir_trmodel + "Validation_values.txt", (loss, mae, lr),
               fmt="%5.5f", header="loss, mae, lr")

    model.load_weights(dir_trmodel + "cp.weights.h5")
    model.save(dir_trmodel + "model.keras")

    with open(dir_trmodel + "history.p", "wb") as f:
        pickle.dump(model.history.history, f)

    with open(dir_trmodel + "report_model.txt", "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n"))

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred      = model.predict(x_test).reshape(len(y_test),)
    yr_pred     = np.array([round(v, 1) for v in y_pred])
    pred_error  = yr_pred - y_test
    abs_error   = np.abs(pred_error)

    with open(dir_pred + "Results_Magnitude.dat", "w") as f:
        f.write("Magnitude, Predicted Mag\n")
        for i in range(len(yr_pred)):
            f.write(f"{y_test[i]} {yr_pred[i]}\n")

    np.savetxt(dir_pred + "Predict_Eval.txt",
               (np.mean(abs_error), np.min(abs_error), np.max(abs_error),
                np.std(pred_error), np.sqrt((pred_error**2).mean())),
               fmt="%10.5f",
               header="mean_error, min_error, max_error, std_error, rms_error")

    volume.commit()  # persist everything to volume

    print(f"\n✅ Done!  loss={loss:.5f}  mae={mae:.5f}  lr={lr:.5f}")
    return {"loss": float(loss), "mae": float(mae), "lr": float(lr)}


# ── 4. Upload local data + run training + download results ────────────────────
@app.local_entrypoint()
def main():
    import pathlib

    nst, nt  = 7, 501                          # ← change here if needed
    case_nm  = f"GNSS_M{nst}S_{nt}"
    project_root = pathlib.Path(__file__).resolve().parents[1]
    local_data = project_root / f"data/processed/tensors/{case_nm}"

    # Step 1: upload .npy files to volume
    print(f"Uploading {case_nm} data to Modal volume...")
    for npy_file in local_data.glob("*.npy"):
        remote_path = f"/data/{case_nm}/{npy_file.name}"
        with npy_file.open("rb") as f:
            volume.put_file(f, remote_path)
        print(f"  ✅ Uploaded {npy_file.name}")
    volume.commit()
    print("Upload complete.\n")

    # Step 2: run training on GPU
    result = train.remote(nst=nst, nt=nt, nc=3)
    print("\nFinal metrics:", result)

    # Step 3: download results back to local ./results/
    print("\nDownloading results...")
    for entry in volume.listdir("/results", recursive=True):
        local_path = project_root / entry.path.lstrip("/")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not entry.is_dir:
            with local_path.open("wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
    print("✅ Results saved to ./results/")