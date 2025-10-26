# Real-time Face Detection with Emgu CV

A simple C# Windows app for real-time face detection using Emgu CV (OpenCV for .NET).

## Features

- Webcam video capture
- Face detection (Haar cascade)
- Face rectangles drawn on video
- Optional LBPH face recognition

## Requirements

- .NET 8.0+
- Windows 10 or 11
- Webcam

## Quick start

1. Restore packages  
   `dotnet restore`

2. Download `haarcascade_frontalface_default.xml` and place in the project folder:
   - [XML link](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)

3. Build  
   `dotnet build`
   
4. Run  
   `dotnet run`
   or run the generated `.exe`

## Usage

- Launch app
- Click "Start Camera"
- Faces are outlined in blue
- (Optional) Enable face recognition checkbox
- Click "Stop Camera" to stop

## Training guide

### Preparing training data

1. Create a folder named `training_data` in the project directory.
2. Inside `training_data`, create a subfolder for each person you want to recognize. Name each subfolder after the person's name or label (e.g., `training_data/Alice`).
3. Add several clear face images for each person inside their respective folders. Name them however you like (e.g., `image1.jpg`, `img2.png`).

### Training the recognizer

- Run the application. If it supports auto-training, the face recognizer will use images in these folders to train.
- Otherwise, consult your `TrainFaceRecognizer.cs` or related code for any manual "Train" actions.

## Troubleshooting

- Webcam not working? Try a different device index or check connection.
- Cascade file error? Confirm XML is present and correct size.

## Project structure
- Core code: `MainForm.cs`, `TrainFaceRecognizer.cs`
- Model: `haarcascade_frontalface_default.xml`
- Build/run: `BuildAndRun.bat`
