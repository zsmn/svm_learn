#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;
int main(int, char**)
{
    // Set up training data
    //int labels[4] = {1, -1, -1, -1};
    //float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };

    int *labels = NULL;
    float **trainingData = NULL;
    int row = 0;
    int qtParameters = 2;
    FILE *input_dataset = fopen("train_dataset.txt", "a+");

    while(!feof(input_dataset)){
        trainingData = (float **) realloc(trainingData, (row + 1) * sizeof(float *));
        labels = (int *) realloc(labels, (row + 1) * sizeof(int));
        trainingData[row] = (float *) malloc(qtParameters * sizeof(float));

        for(int x = 0; x < qtParameters; x++){
            fscanf(input_dataset, "%f", &trainingData[row][x]);
        }

        fscanf(input_dataset, "%d", &labels[row]);
        row++;
    }

    printf("%d %d\n", row, qtParameters);
    for(int x = 0; x < row; x++){
        for(int y = 0; y < qtParameters; y++){
            printf("%f ", trainingData[x][y]);
        }
        printf("%d\n", labels[x]);
    }

    Mat trainingDataMat(row, qtParameters, CV_32F, trainingData);
    Mat labelsMat(row, 1, CV_32SC1, labels);
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);
    // Show the decision regions given by the SVM
    Vec3b green(0,255,0), blue(255,0,0);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
    }
    // Show the training data
    int thickness = -1;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness );
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness );
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness );
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness );
    // Show support vectors
    thickness = 2;
    Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; i++)
    {
        const float* v = sv.ptr<float>(i);
        circle(image,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thickness);
    }
    imwrite("result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    svm->save("output.xml");
    waitKey();
    return 0;
}