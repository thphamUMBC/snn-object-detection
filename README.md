# SNN Object Detection — Car & Human Detection

Spiking Neural Network (SNN) for detecting cars and humans in neuromorphic camera frames. Grid-based (YOLO-style) architecture using `snntorch` with Leaky Integrate-and-Fire (LIF) neurons and Poisson rate-coded spike inputs.

## Requirements

```bash
pip install snntorch torch torchvision pillow matplotlib numpy python-dotenv
```

## Project Structure

```
car-detection/
├── code.ipynb              # Main notebook: dataset, network, training, evaluation
├── visualize_bbox.py       # Draw bounding boxes on images for verification
├── .env.example             # Template for environment variables
├── .env                     # Your local data paths (copy from .env.example)
├── dataset/
│   └── ntu_threshold_20/
│       ├── train/           # .jpg images + .txt annotations
│       ├── test/
│       └── valid/
└── README.md
```

## Dataset Format

Each `.jpg` image has a matching `.txt` file. Each line in the txt file:

```
class x_center y_center width height
```

- `class`: 0 = human, 1 = car
- Coordinates are **YOLO-normalized** (0–1), relative to image dimensions
- One line per object; images can have multiple objects
- Some lines contain 8 coordinate values (2 objects concatenated) — handled automatically

## Quick Start

**0. Configure data paths:**
```bash
cp .env.example .env
# Edit .env to set your data paths if different from defaults
```

**1. Verify bounding boxes:**
```bash
python visualize_bbox.py dataset/ntu_threshold_20/train/
```
Opens each image and draws the annotated bounding boxes (red=human, green=car). Saves as `*_annotated.jpg`.

**2. Run training:**
Open `code.ipynb` in Jupyter and run all cells. The notebook:
1. Installs `snntorch` and `python-dotenv`
2. Loads data paths from `.env` (with fallback defaults)
3. Loads and preprocesses the dataset
4. Defines the SNN detection network (fully-connected LIF)
5. Runs a hyperparameter grid search
6. Plots a heatmap of results

## Architecture

### Grid-Based Detection (YOLO-style)

The 416×416 image is divided into a **13×13 grid**. Each grid cell predicts 7 values:

| Channel | Meaning |
|---------|---------|
| 0–1 | Class logits (human, car) |
| 2–5 | Bounding box: `x, y` (offset within cell), `w, h` (relative to grid) |
| 6 | Objectness confidence (0 = no object, 1 = object) |

The cell responsible for an object is determined by where the object's center falls.

### Network: DetectionNet

```
Input: [T, B, 1, 416, 416]
  ↓
Conv2d(1→16,   3×3) + LIF + MaxPool(2)  →  [T, B, 16,  208, 208]
Conv2d(16→32,  3×3) + LIF + MaxPool(2)  →  [T, B, 32,  104, 104]
Conv2d(32→64,  3×3) + LIF + MaxPool(2)  →  [T, B, 64,   52, 52]
Conv2d(64→128, 3×3) + LIF + MaxPool(2)  →  [T, B, 128,  26, 26]
Conv2d(128→256,3×3) + LIF + MaxPool(2)  →  [T, B, 256,  13, 13]
  ↓
[ Conv2d(256→hidden, 1×1) + LIF ]  × layer_num   (1 or 2 extra layers)
  ↓
Conv2d(→7, 1×1) + LIF  →  [T, B, 7, 13, 13]   ← detection head
```

Five 3×3 conv+pool blocks downsample 416→13 spatial resolution. Each 13×13 grid cell maps directly to a 32×32 pixel region in the input image. Convolutions preserve spatial structure — a car pattern learned in one location is recognized everywhere.

### LIF Neuron (Leaky Integrate-and-Fire)

```
membrane[t] = beta × membrane[t−1] + input[t]
if membrane[t] > 1.0:  fire spike, reset membrane
```

- `beta` controls leak rate: lower = faster decay, shorter memory
- Membrane persists across time steps — this is where temporal dynamics live

### Temporal Processing (the SNN difference)

A standard ANN processes one static image. The SNN processes **T time steps** with different input each time:

1. **Spike encoding** (`spikegen.rate`): Poisson spike trains. At each time step, each pixel independently fires a spike with probability proportional to its intensity. Bright pixel (0.9) → spikes ~90% of the time. Dark pixel (0.1) → spikes ~10% of the time.

2. **Forward pass**: The network runs T times. At each time step:
   - A different set of input spikes enters as `[B, 1, 416, 416]`
   - The spike tensor flows through conv layers → LIF → pooling
   - LIF membranes integrate, leak, and fire
   - Membrane state carries forward between time steps

3. **Output**: Membrane potentials of the head LIF layer are stacked across all T time steps: `[T, B, 7, 13, 13]`. Summed over T for the final prediction: `[B, 7, 13, 13]`.

This temporal integration allows the network to accumulate evidence over time — noisy spikes at individual time steps get smoothed through the LIF dynamics.

## Loss Function

Four components, each applied to different cells:

| Component | Which cells | Weight | Purpose |
|-----------|------------|--------|---------|
| Class MSE | Object cells only | 1.0 | Match predicted class to ground truth |
| Bbox MSE | Object cells only | 5.0 | Match predicted box coordinates |
| Object confidence | Object cells only | 1.0 | Push confidence → 1.0 |
| No-object confidence | Empty cells only | 0.5 | Push confidence → 0.0 |

The lower weight on no-object confidence (0.5 vs 1.0) prevents the 167+ empty cells from overwhelming the 2–3 occupied cells per image.

## Training

For each hyperparameter configuration (16 total):

1. Create `DetectionNet` with the given params
2. For 10 epochs:
   - Spike-encode each image batch into T time steps
   - Forward pass through the SNN
   - Sum membrane potentials over time
   - Compute detection loss
   - Backpropagate, update weights
3. Evaluate F1 score (IoU@0.5) with per-class precision/recall on the test set

### Hyperparameters

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `beta` | 0.85, 0.95 | LIF membrane leak rate |
| `step_num` | 25, 50 | Number of time steps for spike encoding |
| `layer_num` | 1, 2 | Number of extra 1×1 conv + LIF layers after the conv stem |
| `hidden_num` | 64, 128 | Channel count in extra 1×1 conv layers |

Grid search: 2 × 2 × 2 × 2 = **16 configurations**.

## Evaluation

Evaluation uses proper detection metrics with IoU matching:

1. **Decode predictions**: Each cell's output is converted to a detection: class, confidence, and bounding box in normalized image coordinates.
2. **IoU matching**: Predicted boxes are matched to ground truth boxes using Intersection-over-Union (IoU) with a threshold of 0.5.
3. **Per-class metrics**: True positives (TP), false positives (FP), and false negatives (FN) are counted per class.
4. **Aggregate metrics**: Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2×P×R/(P+R).

```
For each test image:
    pred = sum of membrane potentials over T steps  → [7, 13, 13]
    Decode predicted boxes: find cells where confidence > threshold
    Decode ground truth boxes from target tensor
    Match predictions to GT using IoU > 0.5 (greedy, by confidence)
    Compute TP/FP/FN per class → Precision, Recall, F1
```

The old "class accuracy on occupied cells" metric was misleading because it only checked whether the predicted class in a pre-assigned cell matched the ground truth class — it didn't verify the box was actually at the right location. IoU-based evaluation measures whether the network can correctly **both** classify **and** localize objects.

## Edge Cases

- **Multiple objects in the same grid cell**: Only the first is kept (the second is skipped). With 169 cells and typically 2–3 objects per image, collisions are rare.
- **Images with no objects**: Skipped during dataset creation (not included in training).
- **Objects spanning multiple cells**: The object is assigned to the single cell containing its center point.
