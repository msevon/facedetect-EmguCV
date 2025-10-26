using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace FaceDetect.Services
{
    /// <summary>
    /// Service for detecting faces in images using Haar Cascade classifier
    /// </summary>
    public class FaceDetector
    {
        private CascadeClassifier _faceCascade;
        private const string CascadePath = "haarcascade_frontalface_default.xml";

        public FaceDetector()
        {
            _faceCascade = new CascadeClassifier(CascadePath);
        }

        /// <summary>
        /// Detect faces in a grayscale image
        /// </summary>
        /// <param name="grayImage">Grayscale input image</param>
        /// <returns>Array of rectangles containing detected faces</returns>
        public Rectangle[] DetectFaces(Mat grayImage)
        {
            if (grayImage == null || grayImage.IsEmpty)
                return Array.Empty<Rectangle>();

            return _faceCascade.DetectMultiScale(
                grayImage,
                1.1,   // Scale factor
                3,     // Minimum neighbors
                new Size(30, 30), // Minimum size
                Size.Empty        // Maximum size
            );
        }

        /// <summary>
        /// Convert image to grayscale and apply histogram equalization
        /// </summary>
        public Mat PreprocessImage(Mat colorImage)
        {
            Mat grayImage = new Mat();
            CvInvoke.CvtColor(colorImage, grayImage, ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(grayImage, grayImage);
            return grayImage;
        }

        /// <summary>
        /// Draw rectangles around detected faces
        /// </summary>
        public void DrawFaces(Mat frame, Rectangle[] faces)
        {
            foreach (Rectangle face in faces)
            {
                CvInvoke.Rectangle(frame, face, new MCvScalar(255, 0, 0), 3);
            }
        }

        public void Dispose()
        {
            _faceCascade?.Dispose();
        }
    }
}

