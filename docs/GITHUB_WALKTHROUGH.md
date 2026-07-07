# 🚶‍♂️ Veilshift GitHub Walkthrough

Welcome to **Veilshift (QT-2.23)**! 

This guide is meant for users, developers, and scientists who want to get up and running with the Veilshift pipeline quickly. The goal of this application is to take raw, multi-sensor data, mathematically select the best features, and fuse them together to create a powerful, accurate, and completely scientifically honest detector.

---

## ⚙️ Running the Application

Veilshift is built using a modern, interactive web interface. To start the application, simply navigate to the root directory of the project in your terminal and run:

```bash
streamlit run app/main.py
```
*(If your entry point is different, run the corresponding python/streamlit script).*

A browser window will automatically open and drop you into the Veilshift dashboard.

---

## 🎛️ Mission Control

The **Mission Control** tab is your home base. 
Here, you don't need to manually string together Python scripts or worry about training vs. testing splits. The orchestrator handles everything.

### 1. Configure Your Sensors
Before you run an experiment, you can tweak the physics parameters for the built-in sensors (Radar, Thermal, and Acoustic).
- Adjust the **Signal-to-Noise Ratio (SNR)** to make the simulation harder or easier.
- Adjust the environmental clutter and base frequencies.

### 2. Run the Experiment
Click the **"Run Experiment"** button.
Behind the scenes, Veilshift will:
1. Boot up the physics engines and generate realistic sensor data based on your configuration.
2. Segment the data securely into a strict 5-fold cross-validation loop.
3. Automatically determine the best features using a Quantum Unconstrained Binary Optimization (QUBO) algorithm.
4. Train specialized Logistic Regression models for each sensor.
5. Weigh each sensor's predictions and fuse them.
6. Calculate a strict, uncheated final score based purely on data the models *never* saw during training (Out-Of-Fold data).

### 3. Read the Results
Once the progress bar completes, you will see a unified metrics dashboard.
- **AUC (Area Under Curve):** The primary metric indicating how well the fusion engine separates signals from noise.
- **Detection Rate (Pd) & False Alarm Rate (FAR):** The operational performance of the thresholded model.
- **Bootstrap Confidence Intervals:** Shows the statistical reliability of the score.

---

## 🔬 Deeper Inspection

Veilshift doesn't just give you a number; it lets you deeply inspect *why* it made its decisions.

### 📊 Feature Space
Navigate here to see exactly which features the QUBO algorithm decided were mathematically relevant, and how frequently they were selected across different folds.

### ⚖️ Fusion Weights
Curious why Radar was trusted more than Thermal in a specific run? The Fusion tab shows the calculated "trust weights" applied to each sensor's predictions. You can see how the optimizer shifted trust based on the SNR you configured.

### 📈 Baselines
To prove that fusing sensors is actually better than just using one, check the Baselines tab. It rigorously compares the Fused model against a Radar-Only, Thermal-Only, and Acoustic-Only model.

---

## 🛠️ Modifying the Code

Veilshift is highly modular. If you are a developer looking to swap out a classifier or tweak a physics engine:
- **`science/`** contains all the mathematical and physical logic. Each module (Radar, Thermal, Acoustic, QUBO, Fusion) is completely decoupled.
- **`app/`** contains the state management and the primary pipeline orchestrator.
- **`docs/`** contains the deep-dive theory and mathematical architectures explaining the pipeline. 

Enjoy exploring Veilshift!
