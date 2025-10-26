using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace FaceDetect.Services
{
    /// <summary>
    /// Service for training the face recognizer with labeled images
    /// </summary>
    public class FaceTrainer
    {
        private const string TrainingDataPath = "training_data";
        private const string ModelPath = "face_recognizer_model.yml";
        private const string CascadePath = "haarcascade_frontalface_default.xml";
        private const string NamesMapPath = "face_names_map.json";

        /// <summary>
        /// Train the face recognizer using images from training_data folder
        /// </summary>
        public void Train()
        {
            // Validate training data exists
            if (!Directory.Exists(TrainingDataPath))
            {
                throw new Exception($"Training data folder '{TrainingDataPath}' not found. " +
                    "Create a 'training_data' folder and add subfolders for each person with their photos.");
            }

            string[] personDirs = Directory.GetDirectories(TrainingDataPath);
            if (personDirs.Length == 0)
            {
                throw new Exception($"No person directories found in '{TrainingDataPath}'. " +
                    "Create subfolders for each person (e.g., 'training_data/Alice/', 'training_data/Bob/').");
            }

            // Initialize face detector
            CascadeClassifier faceCascade = new CascadeClassifier(CascadePath);
            
            List<Mat> faces = new List<Mat>();
            List<int> labels = new List<int>();
            Dictionary<int, string> labelToName = new Dictionary<int, string>();

            int label = 0;
            foreach (string personDir in personDirs)
            {
                string personName = Path.GetFileName(personDir);
                
                string[] imageFiles = Directory.GetFiles(personDir)
                    .Where(f => f.ToLower().EndsWith(".jpg") || 
                               f.ToLower().EndsWith(".jpeg") || 
                               f.ToLower().EndsWith(".png") ||
                               f.ToLower().EndsWith(".bmp"))
                    .ToArray();

                if (imageFiles.Length == 0)
                    continue;

                // Store the mapping of label to person name
                labelToName[label] = personName;

                // Process each image
                foreach (string imagePath in imageFiles)
                {
                    try
                    {
                        Mat colorImage = CvInvoke.Imread(imagePath, ImreadModes.Color);
                        
                        if (colorImage.IsEmpty)
                            continue;

                        // Convert to grayscale
                        Mat grayImage = new Mat();
                        CvInvoke.CvtColor(colorImage, grayImage, ColorConversion.Bgr2Gray);
                        CvInvoke.EqualizeHist(grayImage, grayImage);

                        // Detect face
                        Rectangle[] detectedFaces = faceCascade.DetectMultiScale(
                            grayImage,
                            1.1, 3, new Size(30, 30), Size.Empty
                        );

                        if (detectedFaces.Length > 0)
                        {
                            Rectangle face = detectedFaces[0];
                            Mat faceROI = new Mat(grayImage, face);
                            
                            // Resize to standard size (larger size = better recognition)
                            Mat resizedFace = new Mat();
                            CvInvoke.Resize(faceROI, resizedFace, new Size(150, 150));
                            
                            faces.Add(resizedFace);
                            labels.Add(label);
                        }

                        colorImage.Dispose();
                        grayImage.Dispose();
                    }
                    catch
                    {
                        // Skip unprocessable images
                    }
                }
                label++;
            }

            if (faces.Count == 0)
            {
                throw new Exception("No faces were extracted from training images. " +
                    "Make sure the photos contain clear, visible faces.");
            }

            // Train the recognizer
            LBPHFaceRecognizer recognizer = new LBPHFaceRecognizer();
            recognizer.Train(faces.ToArray(), labels.ToArray());
            recognizer.Write(ModelPath);

            // Save the label-to-name mapping
            string jsonString = JsonSerializer.Serialize(labelToName, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(NamesMapPath, jsonString);

            // Cleanup
            foreach (Mat face in faces)
                face.Dispose();
            
            faceCascade.Dispose();
            recognizer.Dispose();
        }
    }
}

