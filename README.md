# Overview

An end-to-end Edge AI pipeline that deploys a custom **TinyCNN** onto the **Grove Vision AI Module V2**. This system recognises hand gestures in real-time to play/pause music, achieving low-latency inference in a constrained hardware environment. 

The implementation focuses on maximising compute efficiency on the **Arm Ethos-U55 NPU**, through careful model design and quantization. 

### Key Features
* **Edge-Optimized Architecture:** Custom CNN designed specifically for the constraints of the Himax WiseEye2.
* **Quantization Pipeline:** Implementation of **INT8 Post-Training Quantization** to reduce model size and enable hardware acceleration.
* **Cross-Platform Interoperability:** Utilization of **ONNX** and **TensorFlow Lite** for seamless model translation.
* **Low-Latency Inference:** Real-time processing performed entirely on-device for enhanced privacy and speed.

## Model Architecture
The core is a **TinyCNN** optimized for $96 \times 96$ grayscale input to minimize memory bandwidth:
* **Feature Extraction:** 3 Convolutional layers using strided convolutions for efficient downsampling.
* **Dimensionality Reduction:** Adaptive Average Pooling to reduce parameter count.
* **Generalization & Regularization:** 
    * **Heavy Dropout ($p=0.5$):** Strategically applied to prevent co-adaptation of features in a small-parameter space.
    * **Data Augmentation:** Implemented during training to improve robustness against varying lighting and hand orientations.
    * **Early Stopping** Training conducted with validated early stopping to prevent overfitting, improving generalisation in deployment

## Tech Stack
* **Deep Learning:** PyTorch, TensorFlow Lite
* **Model Exchange:** ONNX (Open Neural Network Exchange)
* **Hardware:** Grove Vision AI Module V2 (Himax WiseEye2 / Arm Ethos-U55)
* **Compiler:** Arm Vela
* **Languages:** Python, PowerShell

## Project Structure
```text
├── deployment/          # Firmware and xmodem flash scripts for Grove V2
├── models/              # PyTorch CNN -> .pth -> .onnx -> .tflite -> .vela
├── notebooks/           # Training experiments and CNN_v3 design
├── scripts/             # Core Logic
│   ├── export.py        # PyTorch to ONNX conversion
│   ├── quantize.py      # INT8 Post-Training Quantization
│   └── music_control.py # Host-side logic for media interaction
└── requirements/        # Modular dependencies
```

## Reproducibility

To replicate the deployment of the TinyCNN onto the Grove Vision AI V2, follow the conversion chain below. Each stage requires a specific environment found in the `requirements/` directory.

### 1. Hardware Requirements
* **Seeed Studio Grove Vision AI Module V2** (Himax WiseEye2 / Arm Ethos-U55).
* **USB-C Cable** for flashing and serial communication.

### 2. Software Conversion Pipeline

The model undergoes a multi-stage transformation to move from PyTorch to hardware-appropriate application.

| Stage | Input | Output | Environment | Command / Script |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | Raw Data | `.pth` | `requirements_training.txt` | `notebooks/CNN_v3.ipynb` |
| **Export** | `.pth` | `.onnx` | `requirements_training.txt` | `python scripts/export.py` |
| **Translation** | `.onnx` | SavedModel (`.tf`) | `requirements_onnx_tf.txt` | `./scripts/onnx_to_tf.ps1` |
| **Quantization**| `.tf` | `model_int8.tflite`| `requirements_quantize.txt` | `python scripts/quantize.py` |
| **Vela Model Compile**| `.tflite`| `model_vela.tflite`| www.edgeimpulse.com | `vela model_int8.tflite` |

### 3. Setup Instructions

#### **A. Training & Export**
1. Install the training dependencies: `pip install -r requirements/requirements_training.txt`.
2. Execute the `notebooks/CNN_v3.ipynb` notebook to train the model and generate the `models/best_model.pth` weights.
3. Convert the weights to ONNX format:
   
```powershell
   python scripts/export.py
```

#### **B. ONNX to TensorFlow**
Because the Grove V2 requires a TFLite flatbuffer, we first translate the ONNX graph into a TensorFlow SavedModel:

1. Install: `pip install -r requirements/requirements_onnx_tf.txt`.
2. Run the PowerShell translation script:
   ```powershell
   ./scripts/onnx_to_tf.ps1

#### **C. INT8 Post-Training Quantization**
To utilize the **Arm Ethos-U55 NPU**, the model must be fully quantized to INT8. This step ensures all weights and activations are mapped to 8-bit integers:

1. Install: `pip install -r requirements/requirements_quantize.txt`.
2. Run the quantization script:
   ```powershell
   python scripts/quantize.py

#### **D. Compilation & Deployment**
1. **Model Compilation:** Upload the `model_int8.tflite` to (https://edgeimpulse.com) to compile it for the **Arm Ethos-U55** NPU, resulting in the `model_vela.tflite` file.
2. **Device Flashing:** To flash the firmware and the compiled model to the Grove Vision AI V2, use the official SenseCraft github (https://github.com/Seeed-Studio/SenseCraft-Web-Toolkit) or the official **XMODEM Python scripts** provided by Seeed Studio.
3. **Host Execution:** Once the hardware is flashed, install the host-side dependencies and run the control logic:
   ```powershell
   pip install -r requirements/requirements_music_control.txt
   python scripts/music_control.py