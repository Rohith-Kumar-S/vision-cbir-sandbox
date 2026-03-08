/*
  Rohith Kumar Senthil Kumar
  Spring 2024
  CS 5330

  Feature extraction functions for histogram features, texture features, and DNN features.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "csv_util/csv_util.h"
#include "filters.h"

std::vector<float> makeCenterPatchFeature(cv::Mat image)
{

    // Extract a 7x7 patch from the center of the image and flatten it into a 1D vector.

    int rows = image.rows;
    int cols = image.cols;
    std::vector<float> imgData;

    int yStart = (rows / 2) - 3;
    int xStart = (cols / 2) - 3;

    cv::Rect centerPatchRect(xStart, yStart, 7, 7);
    cv::Mat centerPatch = image(centerPatchRect);

    for (int i = 0; i < centerPatch.rows; i++)
    {
        cv::Vec3b *centerPatchRow = centerPatch.ptr<cv::Vec3b>(i);
        for (int j = 0; j < centerPatch.cols; j++)
        {
            imgData.push_back(centerPatchRow[j][0]);
            imgData.push_back(centerPatchRow[j][1]);
            imgData.push_back(centerPatchRow[j][2]);
        }
    }
    return imgData;
}

std::vector<float> makeRGHist(cv::Mat image, const int histsize)
{
    // 2D histogram for r and g chromaticity channels
    cv::Mat hist;
    std::vector<float> imgData;
    hist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32FC1);

    for (int i = 0; i < image.rows; i++)
    {
        cv::Vec3b *imgRow = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image.cols; j++)
        {
            float B = imgRow[j][0];
            float G = imgRow[j][1];
            float R = imgRow[j][2];

            // Chromaticity channels
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0;
            float r = R / divisor;
            float g = G / divisor;

            // Index computation
            int rIndex = (r * (histsize - 1)) + 0.5;
            int gIndex = (g * (histsize - 1)) + 0.5;

            hist.at<float>(rIndex, gIndex)++;
        }
    }
    hist /= (image.rows * image.cols);
    for (int i = 0; i < hist.rows; i++)
    {
        float *histRow = hist.ptr<float>(i);
        for (int j = 0; j < hist.cols; j++)
        {
            imgData.push_back(histRow[j]);
        }
    }
    return imgData;
}

std::vector<float> makeRGBHist(cv::Mat image, const int histsize)
{
    // 3D histogram for RGB channels
    std::vector<float> imgData;
    int sizes[3] = {histsize, histsize, histsize};
    cv::Mat hist(3, sizes, CV_32F, cv::Scalar(0));

    const float rScale = static_cast<float>(histsize) / 256.0f;
    const float gScale = static_cast<float>(histsize) / 256.0f;
    const float bScale = static_cast<float>(histsize) / 256.0f;

    for (int y = 0; y < image.rows; ++y)
    {
        const cv::Vec3b *rowPtr = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; ++x)
        {
            const uchar B = rowPtr[x][0];
            const uchar G = rowPtr[x][1];
            const uchar R = rowPtr[x][2];

            int rBinIdx = static_cast<int>(R * rScale);
            int gBinIdx = static_cast<int>(G * gScale);
            int bBinIdx = static_cast<int>(B * bScale);

            // Clamp
            rBinIdx = std::min(std::max(rBinIdx, 0), histsize - 1);
            gBinIdx = std::min(std::max(gBinIdx, 0), histsize - 1);
            bBinIdx = std::min(std::max(bBinIdx, 0), histsize - 1);

            int idx[3] = {rBinIdx, gBinIdx, bBinIdx};
            hist.at<float>(idx) += 1.0f;
        }
    }

    const float divisor = static_cast<float>(image.rows * image.cols);
    if (divisor > 0.0f)
        hist /= divisor;

    // Iterate over the 3D histogram and flatten it into a 1D vector
    for (int r = 0; r < histsize; ++r)
    {
        for (int g = 0; g < histsize; ++g)
        {
            for (int b = 0; b < histsize; ++b)
            {
                int idx[3] = {r, g, b};
                imgData.push_back(hist.at<float>(idx));
            }
        }
    }
    return imgData;
}


std::vector<float> makeTextureHist(cv::Mat image, const int histsize)
{
    // 2D histogram for magnitude and orientation of gradients

    cv::Mat sobelx, sobely, magnitude, orientation, hist, magnitudeGray, orientationGray;
    std::vector<float> imgData;
    hist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32FC1);

    Filters::sobelX3x3Gray(image, sobelx);
    Filters::sobelY3x3Gray(image, sobely);

    Filters::magnitude(sobelx, sobely, magnitude);

    Filters::orientation(sobelx, sobely, orientation);
    double maxMagValue = 0, maxOrientValue = 0;
    cv::minMaxLoc(magnitude, nullptr, &maxMagValue);
    cv::minMaxLoc(orientation, nullptr, &maxOrientValue);

    if (maxMagValue < 1e-6)
        maxMagValue = 1.0;
    if (maxOrientValue < 1e-6)
        maxOrientValue = 1.0;

    for (int i = 0; i < image.rows; i++)
    {
        short *magnitudeGrayRow = magnitude.ptr<short>(i);
        float *orientationGrayRow = orientation.ptr<float>(i);
        for (int j = 0; j < image.cols; j++)
        {
            // Chromaticity channels
            float magnitudeNorm = static_cast<float>(magnitudeGrayRow[j]) / static_cast<float>(maxMagValue);
            float orientationNorm = (orientationGrayRow[j] + CV_PI) / (2 * CV_PI);

            // Index computation
            int mIndex = (magnitudeNorm * (histsize - 1)) + 0.5;
            int oIndex = (orientationNorm * (histsize - 1)) + 0.5;

            hist.at<float>(mIndex, oIndex)++;
        }
    }
    hist /= (image.rows * image.cols);
    for (int i = 0; i < hist.rows; i++)
    {
        float *histRow = hist.ptr<float>(i);
        for (int j = 0; j < hist.cols; j++)
        {
            imgData.push_back(histRow[j]);
        }
    }
    return imgData;
}

std::vector<float> makeS5S5Hist(cv::Mat image, cv::Mat filter, int histSize = 16)
{
    // Apply S5S5 filter to the image, compute the absolute response, and build a histogram of the responses.

    cv::Mat gray, lawsResponse;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    cv::filter2D(gray, lawsResponse, CV_32F, filter);
    lawsResponse = cv::abs(lawsResponse);

    // Min max normalization
    double minVal, maxVal;
    cv::minMaxLoc(lawsResponse, &minVal, &maxVal);
    if (maxVal > minVal)
    {
        lawsResponse = (lawsResponse - minVal) / (maxVal - minVal);
    }

    // Build histogram
    std::vector<float> imgData(histSize, 0.0f);
    for (int i = 0; i < lawsResponse.rows; i++)
    {
        float *row = lawsResponse.ptr<float>(i);
        for (int j = 0; j < lawsResponse.cols; j++)
        {
            int binIdx = (int)(row[j] * (histSize - 1) + 0.5f);
            binIdx = std::min(histSize - 1, binIdx);
            imgData[binIdx]++;
        }
    }

    // Normalize
    float divisor = lawsResponse.rows * lawsResponse.cols;
    for (auto &val : imgData)
        val /= divisor;

    return imgData;
}

void computeS5S5LawsFilter(cv::Mat &filter)
{
    // Create S5S5 filter in constructor
    std::vector<float> S5 = {-1, 0, 2, 0, -1};

    // Initialize the filter
    filter = cv::Mat(5, 5, CV_32F);
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            filter.at<float>(i, j) = S5[i] * S5[j];
        }
    }
}

int makeAndStoreHist(int patchType = 0)
{
    // Read all image filenames from the "olympus" directory and compute features for each image, storing them in CSV files.
    std::vector<std::string> filenames;
    cv::glob("olympus/*.jpg", filenames, false);
    cv::Mat hist, image, filter;
    const int histsize = 16;
    computeS5S5LawsFilter(filter);
    for (int k = 0; k < filenames.size(); k++)
    {

        image = cv::imread(filenames[k]);

        if (image.data == NULL)
        {
            printf("error: unable to read filename %s\n", filenames[k].c_str());
            return (-2);
        }

        std::vector<float> topHalf, bottomHalf, rgHist, textureHist, centerPatchFeature, s5s5Hist, imgData, rgbHist;
        rgHist = makeRGHist(image, histsize);
        topHalf = makeRGHist(image(cv::Rect(0, 0, image.cols, image.rows / 2)), histsize);
        bottomHalf = makeRGHist(image(cv::Rect(0, image.rows / 2, image.cols, image.rows / 2)), histsize);
        textureHist = makeTextureHist(image, histsize);
        centerPatchFeature = makeCenterPatchFeature(image);
        rgbHist = makeRGBHist(image, 8);
        s5s5Hist = makeS5S5Hist(image, filter, histsize);
        int reset = (k == 0) ? 1 : 0;
        append_image_data_csv("hm_features.csv", const_cast<char *>(filenames[k].c_str()), rgHist, reset);
        append_image_data_csv("hm_topHalf.csv", const_cast<char *>(filenames[k].c_str()), topHalf, reset);
        append_image_data_csv("hm_bottomHalf.csv", const_cast<char *>(filenames[k].c_str()), bottomHalf, reset);
        append_image_data_csv("hm_textures.csv", const_cast<char *>(filenames[k].c_str()), textureHist, reset);
        append_image_data_csv("cp_features.csv", const_cast<char *>(filenames[k].c_str()), centerPatchFeature, reset);
        append_image_data_csv("hm_rgb_features.csv", const_cast<char *>(filenames[k].c_str()), rgbHist, reset);
        append_image_data_csv("hm_s5s5_features.csv", const_cast<char *>(filenames[k].c_str()), s5s5Hist, reset);
    }
    return 0;
}

// int main()
// {
//     makeAndStoreHist();
//     return 0;
// }