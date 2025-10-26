using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace FaceDetect.Services
{
    /// <summary>
    /// Service for recognizing faces using LBPH algorithm
    /// </summary>
    public class FaceRecognizerService
    {
        private LBPHFaceRecognizer? _recognizer;
        private Dictionary<int, string> _labelToNameMap = new Dictionary<int, string>();
        private const string ModelPath = "face_recognizer_model.yml";
        private const string NamesMapPath = "face_names_map.json";

        public bool IsModelLoaded { get; private set; }

        public FaceRecognizerService()
        {
            LoadModel();
        }

        /// <summary>
        /// Load the trained recognition model if it exists
        /// </summary>
        public void LoadModel()
        {
            try
            {
                if (File.Exists(ModelPath))
                {
                    _recognizer = new LBPHFaceRecognizer();
                    _recognizer.Read(ModelPath);
                    IsModelLoaded = true;
                    
                    // Load the label-to-name mapping
                    if (File.Exists(NamesMapPath))
                    {
                        string jsonString = File.ReadAllText(NamesMapPath);
                        var map = JsonSerializer.Deserialize<Dictionary<string, string>>(jsonString);
                        if (map != null)
                        {
                            _labelToNameMap = map.ToDictionary(
                                kvp => int.Parse(kvp.Key), 
                                kvp => kvp.Value
                            );
                        }
                    }
                }
                else
                {
                    IsModelLoaded = false;
                }
            }
            catch
            {
                IsModelLoaded = false;
            }
        }

        /// <summary>
        /// Recognize a face from the face region
        /// </summary>
        /// <param name="faceRegion">Grayscale face image</param>
        /// <returns>Tuple of (label, confidence, personName)</returns>
        public (int label, double confidence, string personName) RecognizeFace(Mat faceRegion)
        {
            if (_recognizer == null || faceRegion == null || faceRegion.IsEmpty)
            {
                return (-1, double.MaxValue, "Unknown");
            }

            try
            {
                var result = _recognizer.Predict(faceRegion);
                
                // For LBPH: lower distance = better match
                string personName;
                if (result.Label >= 0 && _labelToNameMap.ContainsKey(result.Label))
                {
                    personName = _labelToNameMap[result.Label];
                }
                else
                {
                    personName = "Unknown";
                }
                
                return (result.Label, result.Distance, personName);
            }
            catch
            {
                return (-1, double.MaxValue, "Unknown");
            }
        }

        /// <summary>
        /// Draw recognition label on the frame
        /// </summary>
        public void DrawRecognitionLabel(Mat frame, Rectangle faceRect, (int label, double confidence, string personName) recognitionResult)
        {
            Point textLocation = new Point(faceRect.X, faceRect.Y - 10);
            MCvScalar textColor;
            string displayText;

            // For LBPH: lower distance = better match
            // Typical thresholds:
            // 0-35 = Excellent match
            // 35-65 = Good match
            // 65-100 = Acceptable match
            // 100+ = Unknown
            
            if (recognitionResult.confidence < 35)
            {
                // Excellent match - high confidence
                displayText = $"{recognitionResult.personName} ({recognitionResult.confidence:F0})";
                textColor = new MCvScalar(0, 255, 0); // Green
            }
            else if (recognitionResult.confidence < 65)
            {
                // Good match - medium confidence
                displayText = $"{recognitionResult.personName} ({recognitionResult.confidence:F0})";
                textColor = new MCvScalar(255, 255, 0); // Yellow
            }
            else if (recognitionResult.confidence < 100)
            {
                // Low confidence - show but in orange
                displayText = $"{recognitionResult.personName}? ({recognitionResult.confidence:F0})";
                textColor = new MCvScalar(0, 165, 255); // Orange
            }
            else
            {
                // Unknown - not recognized
                displayText = $"Unknown ({recognitionResult.confidence:F0})";
                textColor = new MCvScalar(0, 0, 255); // Red
            }

            CvInvoke.PutText(
                frame,
                displayText,
                textLocation,
                FontFace.HersheySimplex,
                0.5,
                textColor,
                2
            );
        }

        public void Dispose()
        {
            _recognizer?.Dispose();
        }
    }
}

