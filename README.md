# Bike Fit Analyzer

A Python tool for optimizing road bike saddle height using computer vision and real-time motion analysis.

## Features

- **Real-time pose detection** using CV-Zone and MediaPipe
- **Automatic knee angle calculation** during pedaling
- **Visual feedback** with color-coded feedback system
- **Maximum tracking** to capture greatest leg extension
- **Reset function** for new measurements after saddle adjustment

## Prerequisites

- Python 3.8 or higher
- Webcam (integrated or external)
- Sufficient space to pedal sideways to the camera

## Installation

1. Clone repository:
```bash
git clone https://github.com/[your-username]/bike-fit-analyzer.git
cd bike-fit-analyzer
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # MacOS/Linux
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Webcam Adjustment

The webcam configuration can be found in the `__init__()` method:

```python
camera_index = 0  # <- ADJUST HERE IF NEEDED
```

**Typical values:**
- MacBook integrated camera: `0` or `1`
- Windows laptop: `0`
- External USB webcam: `1`, `2`, or `3`

### Optimal Angle Range

The optimal knee angle can be adjusted in the `__init__()` method:

```python
self.optimal_angle_min = 140  # Minimum optimal angle
self.optimal_angle_max = 150  # Maximum optimal angle
```

## Usage

1. Start program:
```bash
python bike_fit_analyzer.py
```

2. Position your computer so you can see yourself from the side
3. Press **SPACEBAR** to start the measurement process
4. You have **5 seconds** to get on your bike
5. Pedal normally for **5 seconds** while the system measures
6. After measurement, the maximum angle is locked and displayed with color-coded feedback

### Measurement Process

The program follows a two-phase measurement sequence:
1. **Setup Phase**: Position yourself and your bike sideways to the camera
2. **Countdown (10 seconds)**: Press SPACEBAR and mount your bike
3. **Static Measurement (3 seconds)**: Hold your lowest pedal position
4. **Pause (3 seconds)**: Prepare to start pedaling
5. **Dynamic Measurement (5 seconds)**: Pedal normally while system tracks maximum angle
6. **Results**: Both static and dynamic angles displayed with analysis

### Keyboard Commands

- **`SPACEBAR`** - Start new measurement (5s countdown + 5s measurement)
- **`q`** - Quit program

### Feedback Colors

- **Green**: Saddle height optimal (140-150 degrees)
- **Orange**: Saddle too low (< 140 degrees)
- **Red**: Saddle too high (> 150 degrees)

### Display Information

During operation, the program shows:
- **Current angle**: Real-time knee angle
- **Maximum angle**: Highest angle recorded during pedaling (with moving average filter)
- **Optimal range**: Target angle range for proper saddle height
- **Feedback**: Clear instructions based on your maximum angle
- **State indicator**: Current phase of measurement process

### Accuracy Features

- **Moving average filter**: 10-sample window smooths measurements and eliminates outliers
- **Real-time tracking**: Continuously updates maximum angle during measurement
- **Visual feedback**: Color-coded results for instant understanding

## Scientific Background

The optimal knee extension for cycling is between 140 and 150 degrees:
- **< 140 degrees**: Increased knee stress, inefficient power transfer
- **> 150 degrees**: Overextended position, hip instability
- **140-150 degrees**: Optimal balance between performance and joint protection

### Why Structured Measurement?

The 5-second measurement window ensures:
- Multiple complete pedal rotations are captured
- True maximum extension is recorded (not just a momentary position)
- Consistent results across different cadences
- Eliminates variability from single-point measurements

### Measurement Approach

The system provides:
- **Continuous tracking**: Monitors knee angle throughout pedaling motion
- **Maximum detection**: Captures true extension during active pedaling
- **Enhanced accuracy**: Moving average filtering removes measurement noise and outliers
- **Real-time feedback**: See your current angle while pedaling

## Troubleshooting

### Camera Not Recognized
1. Check if camera is being used by other applications
2. Change the `camera_index` value
3. Ensure camera permissions are granted

### Pose Not Detected
- Ensure good lighting
- Wear contrasting clothing
- Make sure your entire leg is visible in the frame

### MacOS-Specific Issues
- Grant Terminal/Python camera access in System Preferences
- Consider using `opencv-contrib-python` instead of `opencv-python`

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Pull requests are welcome! For major changes, please open an issue first.

## Contact

For questions or issues, please create an [Issue](https://github.com/[your-username]/bike-fit-analyzer/issues).