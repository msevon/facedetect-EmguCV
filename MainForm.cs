using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using FaceDetect.Services;

namespace FaceDetect
{
    /// <summary>
    /// Main form for real-time face detection application
    /// </summary>
    public partial class MainForm : Form
    {
        // Services
        private FaceDetector? _faceDetector;
        private FaceRecognizerService? _faceRecognizerService;
        private FaceTrainer? _faceTrainer;
        
        // Webcam capture
        private VideoCapture? _capture;
        
        // UI components
        private PictureBox _pictureBox = null!;
        private System.Windows.Forms.Timer _timer = null!;
        
        private Label _statusLabel = null!;
        private Label _recognitionStatusLabel = null!;
        private Label _frameCounterLabel = null!;
        
        private Button _startButton = null!;
        private Button _stopButton = null!;
        private CheckBox _enableRecognitionCheckBox = null!;
        private Button _trainButton = null!;
        
        // State
        private int _frameCount = 0;
        private bool _recognitionEnabled = false;

        public MainForm()
        {
            InitializeUI();
            InitializeServices();
        }

        private void InitializeUI()
        {
            this.Text = "Real-Time Face Detection";
            this.Size = new Size(800, 650);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(240, 240, 240);
            
            // Video display
            _pictureBox = new PictureBox
            {
                Location = new Point(10, 50),
                Size = new Size(640, 480),
                BorderStyle = BorderStyle.FixedSingle,
                BackColor = Color.Black
            };
            this.Controls.Add(_pictureBox);
            
            // Status label
            _statusLabel = new Label
            {
                Text = "Status: Stopped",
                Location = new Point(660, 50),
                Size = new Size(130, 20),
                ForeColor = Color.Blue
            };
            this.Controls.Add(_statusLabel);
            
            // Start button
            _startButton = new Button
            {
                Text = "Start Camera",
                Location = new Point(660, 80),
                Size = new Size(130, 35),
                BackColor = Color.FromArgb(76, 175, 80),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat
            };
            _startButton.Click += StartCamera_Click;
            this.Controls.Add(_startButton);
            
            // Stop button
            _stopButton = new Button
            {
                Text = "Stop Camera",
                Location = new Point(660, 125),
                Size = new Size(130, 35),
                BackColor = Color.FromArgb(244, 67, 54),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Enabled = false
            };
            _stopButton.Click += StopCamera_Click;
            this.Controls.Add(_stopButton);
            
            // Recognition checkbox
            _enableRecognitionCheckBox = new CheckBox
            {
                Text = "Enable Face Recognition",
                Location = new Point(660, 180),
                Size = new Size(130, 25),
                Checked = false
            };
            _enableRecognitionCheckBox.CheckedChanged += RecognitionCheckBox_CheckedChanged;
            this.Controls.Add(_enableRecognitionCheckBox);
            
            // Recognition status label
            _recognitionStatusLabel = new Label
            {
                Text = "Recognition: Disabled",
                Location = new Point(660, 210),
                Size = new Size(130, 20),
                ForeColor = Color.Gray,
                Font = new Font("Arial", 8)
            };
            this.Controls.Add(_recognitionStatusLabel);
            
            // Train button
            _trainButton = new Button
            {
                Text = "Train Recognizer",
                Location = new Point(660, 240),
                Size = new Size(130, 35),
                BackColor = Color.FromArgb(33, 150, 243),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat
            };
            _trainButton.Click += TrainButton_Click;
            this.Controls.Add(_trainButton);
            
            // Info label
            var infoLabel = new Label
            {
                Text = "Real-Time Face Detection with Emgu CV\nDetected faces will be highlighted with blue rectangles.",
                Location = new Point(10, 10),
                Size = new Size(640, 35),
                ForeColor = Color.FromArgb(33, 33, 33),
                Font = new Font("Arial", 9)
            };
            this.Controls.Add(infoLabel);
            
            // Frame counter
            _frameCounterLabel = new Label
            {
                Name = "frameCounterLabel",
                Text = "Frames: 0",
                Location = new Point(660, 280),
                Size = new Size(130, 20),
                ForeColor = Color.DarkGreen,
                Font = new Font("Arial", 8)
            };
            this.Controls.Add(_frameCounterLabel);
            
            // Timer for frame capture
            _timer = new System.Windows.Forms.Timer();
            _timer.Interval = 33; // ~30 FPS
            _timer.Tick += Timer_Tick;
        }

        private void InitializeServices()
        {
            try
            {
                _faceDetector = new FaceDetector();
                _faceRecognizerService = new FaceRecognizerService();
                _faceTrainer = new FaceTrainer();
                
                _statusLabel.Text = "Status: Ready";
                _statusLabel.ForeColor = Color.Green;
                UpdateRecognitionStatus();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error initializing services: {ex.Message}", 
                    "Initialization Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                _statusLabel.Text = "Status: Error";
                _statusLabel.ForeColor = Color.Red;
            }
        }

        private void UpdateRecognitionStatus()
        {
            if (_faceRecognizerService?.IsModelLoaded == true)
            {
                _recognitionStatusLabel.Text = "Recognition: Model Ready";
                _recognitionStatusLabel.ForeColor = Color.Green;
            }
            else
            {
                _recognitionStatusLabel.Text = "Recognition: No Model";
                _recognitionStatusLabel.ForeColor = Color.Orange;
            }
        }

        private void StartCamera_Click(object? sender, EventArgs e)
        {
            try
            {
                _capture = new VideoCapture(0);
                
                if (!_capture.IsOpened)
                {
                    MessageBox.Show("Unable to open webcam.", "Camera Error", 
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                _capture.Set(CapProp.FrameWidth, 640);
                _capture.Set(CapProp.FrameHeight, 480);
                
                _timer.Start();
                
                _statusLabel.Text = "Status: Running";
                _statusLabel.ForeColor = Color.Green;
                _startButton.Enabled = false;
                _stopButton.Enabled = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error starting camera: {ex.Message}", 
                    "Camera Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                _statusLabel.Text = "Status: Error";
                _statusLabel.ForeColor = Color.Red;
            }
        }

        private void StopCamera_Click(object? sender, EventArgs e)
        {
            _timer.Stop();
            
            _capture?.Dispose();
            _capture = null;
            
            _pictureBox.Image = null;
            
            _statusLabel.Text = "Status: Stopped";
            _statusLabel.ForeColor = Color.Blue;
            _startButton.Enabled = true;
            _stopButton.Enabled = false;
            
            _frameCount = 0;
            _frameCounterLabel.Text = "Frames: 0";
        }

        private void RecognitionCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _recognitionEnabled = _enableRecognitionCheckBox.Checked;
            
            // Show guidance if recognition is enabled
            if (_recognitionEnabled && _faceRecognizerService != null)
            {
                if (!_faceRecognizerService.IsModelLoaded)
                {
                    MessageBox.Show("No recognition model found!\n\n" +
                        "1. Put photos in training_data/ folder\n" +
                        "2. Each person in their own subfolder\n" +
                        "3. Use 10-20 clear photos per person\n" +
                        "4. Click 'Train Recognizer' button", 
                        "No Model", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    _enableRecognitionCheckBox.Checked = false;
                }
                else
                {
                    MessageBox.Show("Recognition enabled!\n\n" +
                        "Color guide:\n" +
                        "Green (0-35): Excellent match\n" +
                        "Yellow (35-65): Good match\n" +
                        "Orange (65-100): Maybe match\n" +
                        "Red (100+): Unknown",
                        "Recognition Active", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
        }

        private void TrainButton_Click(object? sender, EventArgs e)
        {
            try
            {
                if (_faceTrainer == null)
                {
                    MessageBox.Show("Training service not initialized.", 
                        "Training Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                _trainButton.Enabled = false;
                _trainButton.Text = "Training...";
                
                _faceTrainer.Train();
                
                // Reload the recognizer with new model
                _faceRecognizerService?.Dispose();
                _faceRecognizerService = new FaceRecognizerService();
                
                UpdateRecognitionStatus();
                
                MessageBox.Show("Training completed successfully!\nThe model has been saved.\n\nYou can now enable face recognition.", 
                    "Training Complete", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Training failed: {ex.Message}\n\nMake sure:\n" +
                    "- The 'training_data' folder exists\n" +
                    "- Each person has their own subfolder with photos\n" +
                    "- Photos contain clear face images", 
                    "Training Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                _trainButton.Enabled = true;
                _trainButton.Text = "Train Recognizer";
            }
        }

        private void Timer_Tick(object? sender, EventArgs e)
        {
            if (_capture == null || !_capture.IsOpened || _faceDetector == null)
                return;

            try
            {
                using (Mat frame = _capture.QueryFrame())
                {
                    if (frame != null && !frame.IsEmpty)
                    {
                        ProcessFrame(frame);
                        _frameCount++;
                        _frameCounterLabel.Text = $"Frames: {_frameCount}";
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error processing frame: {ex.Message}");
            }
        }

        private void ProcessFrame(Mat frame)
        {
            if (_faceDetector == null) return;

            // Preprocess image
            Mat grayFrame = _faceDetector.PreprocessImage(frame);
            
            // Detect faces
            Rectangle[] faces = _faceDetector.DetectFaces(grayFrame);
            
            // Draw face rectangles
            _faceDetector.DrawFaces(frame, faces);
            
            // If recognition is enabled, try to identify faces
            if (_recognitionEnabled && _faceRecognizerService != null && _faceRecognizerService.IsModelLoaded)
            {
                foreach (Rectangle face in faces)
                {
                    Mat faceROI = new Mat(grayFrame, face);
                    
                    // Resize to match training size (150x150 for best results)
                    Mat resizedFace = new Mat();
                    CvInvoke.Resize(faceROI, resizedFace, new Size(150, 150));
                    
                    var recognitionResult = _faceRecognizerService.RecognizeFace(resizedFace);
                    _faceRecognizerService.DrawRecognitionLabel(frame, face, recognitionResult);
                    
                    faceROI.Dispose();
                    resizedFace.Dispose();
                }
            }
            
            grayFrame.Dispose();
            
            // Update display
            Bitmap? bmp = frame.ToBitmap();
            if (bmp != null)
            {
                if (_pictureBox.InvokeRequired)
                {
                    _pictureBox.Invoke(new Action(() =>
                    {
                        _pictureBox.Image?.Dispose();
                        _pictureBox.Image = bmp;
                    }));
                }
                else
                {
                    _pictureBox.Image?.Dispose();
                    _pictureBox.Image = bmp;
                }
            }
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            _timer?.Stop();
            _timer?.Dispose();
            
            _capture?.Dispose();
            
            _faceDetector?.Dispose();
            _faceRecognizerService?.Dispose();
            
            base.OnFormClosed(e);
        }
    }

    /// <summary>
    /// Program entry point
    /// </summary>
    class Program
    {
        [STAThread]
        static void Main()
        {
            ApplicationConfiguration.Initialize();
            Application.Run(new MainForm());
        }
    }
}
