/*
  Rohith Kumar Senthil Kumar
  Spring 2024
  CS 5330

  Utility functions for CBIR tasks, including distance computation and patch matching.
*/

#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <random>
#include "csv_util/csv_util.h"
#include "featureExtraction.h"

float computeDistance(std::vector<float> &src, std::vector<float> &target,
                      const std::string &distanceMetric = "l2", int patchMode = 0)
{
    // Compute distance between two feature vectors based on the specified distance metric (L2, L1, Cosine, Intersection).
    float distance = 0.0;

    if (distanceMetric == "cosine")
    {
        cv::Mat srcMat(src);
        cv::Mat targetMat(target);
        float cosSim = srcMat.dot(targetMat) / (cv::norm(srcMat, cv::NORM_L2) * cv::norm(targetMat, cv::NORM_L2));
        return 1.0f - cosSim;
    }

    for (int i = 0; i < src.size(); i++)
    {
        if (distanceMetric == "inf")
            distance = std::max(distance, std::abs(src[i] - target[i]));
        else if (distanceMetric == "l1")
            distance += std::abs(src[i] - target[i]);
        else if (distanceMetric == "inter")
            distance += std::min(src[i], target[i]);
        else
            distance += (src[i] - target[i]) * (src[i] - target[i]);
    }
    if (distanceMetric == "inter")
        distance = 1.0f - distance;
    return distance;
}

void patchMatcher_0(std::vector<cv::Mat> srcImgPatches,
                    std::vector<std::vector<float>>
                        &srcImgHists,
                    std::vector<char *> &imageNames,
                    std::vector<std::vector<std::vector<float>>> &features)
{

    /*  Custom patch matcher that loads pre-computed center patch features
    from a CSV file and computes the center patch feature for each
    image patch, storing them in the provided data structures.*/

    std::vector<std::vector<float>> featureVectors1;
    // load center patch features
    read_image_data_csv("cp_features.csv",
                        imageNames, featureVectors1, 0);

    // n is Number of images
    int n = featureVectors1.size();

    if (features.size() < n)
        features.resize(n);

    // Load the histogram feature vectors
    for (int i = 0; i < n; i++)
    {
        features[i].push_back(featureVectors1[i]);
    }

    for (int patchId = 0; patchId < srcImgPatches.size(); patchId++)
    {
        srcImgHists.push_back(makeCenterPatchFeature(srcImgPatches[patchId]));
    }
}

void patchMatcher_1(std::vector<cv::Mat> srcImgPatches,
                    std::vector<std::vector<float>>
                        &srcImgHists,
                    std::vector<char *> &imageNames,
                    std::vector<std::vector<std::vector<float>>> &features,
                    const int histsize)
{
    /* RG histogram patch matcher  
    that loads pre-computed RG histogram features from a CSV file 
    and computes the RG histogram feature for each image patch, 
    storing them in the provided data structures.
    */

    // lets us to add multiple feture vectors
    std::vector<std::vector<float>> featureVectors1;
    read_image_data_csv("hm_features.csv",
                        imageNames, featureVectors1, 0);

    // n is Number of images
    int n = featureVectors1.size();
    // H is number of histograms per image
    int H = 1;
    // D is number of bins in the histogram
    int D = featureVectors1[0].size();

    // features is a vector of size n, where each element is a vector of H histograms, and each histogram is a vector of D bins.

    if (features.size() < n)
        features.resize(n);

    // Load the histogram feature vectors
    for (int i = 0; i < n; i++)
    {
        features[i].push_back(featureVectors1[i]);
    }

    for (int patchId = 0; patchId < srcImgPatches.size(); patchId++)
    {
        srcImgHists.push_back(makeRGHist(srcImgPatches[patchId], histsize));
    }
}

void patchMatcher_2(std::vector<cv::Mat> srcImgPatches,
                    std::vector<std::vector<float>>
                        &srcImgHists,
                    std::vector<char *> &imageNames,
                    std::vector<std::vector<std::vector<float>>> &features,
                    const int histsize)
{

    /* RGB histogram patch matcher  
    that loads pre-computed RGB histogram features from a CSV file 
    and computes the RGB histogram feature for each image patch, 
    storing them in the provided data structures.
    */

    // lets us to add multiple feture vectors
    std::vector<std::vector<float>> featureVectors1;
    read_image_data_csv("hm_rgb_features.csv",
                        imageNames, featureVectors1, 0);
    // n is Number of images
    int n = featureVectors1.size();

    if (features.size() < n)
        features.resize(n);

    // Load the histogram feature vectors
    for (int i = 0; i < n; i++)
    {
        features[i].push_back(featureVectors1[i]);
    }

    for (int patchId = 0; patchId < srcImgPatches.size(); patchId++)
    {
        srcImgHists.push_back(makeRGBHist(srcImgPatches[patchId], histsize));
    }
}

void patchMatcher_3(std::vector<cv::Mat> srcImgPatches,
                    std::vector<std::vector<float>>
                        &srcImgHists,
                    std::vector<char *> &imageNames,
                    std::vector<std::vector<std::vector<float>>> &features,
                    const int histsize)
{
    /* Custom patch matcher that loads pre-computed top half and bottom half RG histogram features
    from CSV files and computes the RG histogram features for the top half and bottom half of each
    image patch, storing them in the provided data structures.
    */

    // lets us to add multiple feture vectors
    std::vector<std::vector<float>> featureVectors1, featureVectors2;
    std::vector<char *> dummy;
    read_image_data_csv("hm_topHalf.csv",
                        dummy, featureVectors1, 0);
    read_image_data_csv("hm_bottomHalf.csv",
                        imageNames, featureVectors2, 0);

    // n is Number of images
    int n = featureVectors1.size();

    if (features.size() < n)
        features.resize(n);

    // Load the histogram feature vectors
    for (int i = 0; i < n; i++)
    {
        features[i].push_back(featureVectors1[i]);
        features[i].push_back(featureVectors2[i]);
    }

    for (int patchId = 0; patchId < srcImgPatches.size(); patchId++)
    {
        srcImgHists.push_back(makeRGHist(srcImgPatches[patchId], histsize));
    }
}

void patchMatcher_4(std::vector<cv::Mat> srcImgPatches,
                    std::vector<std::vector<float>>
                        &srcImgHists,
                    std::vector<char *> &imageNames,
                    std::vector<std::vector<std::vector<float>>> &features,
                    const int histsize)
{

    /*
    Custom patch matcher that loads pre-computed feature and texture histogram features
    Uses rg histogram for color and magnitude and orientation histogram for texture. */

    // lets us to add multiple feture vectors
    std::vector<std::vector<float>> featureVectors1, featureVectors2;
    std::vector<char *> dummy;
    read_image_data_csv("hm_features.csv",
                        imageNames, featureVectors1, 0);
    read_image_data_csv("hm_textures.csv",
                        dummy, featureVectors2, 0);

    // n is Number of images
    int n = featureVectors1.size();

    if (features.size() < n)
        features.resize(n);

    // Load the histogram feature vectors
    for (int i = 0; i < n; i++)
    {
        features[i].push_back(featureVectors1[i]);
        features[i].push_back(featureVectors2[i]);
    }
    srcImgHists.push_back(makeRGHist(srcImgPatches[0], histsize));
    srcImgHists.push_back(makeTextureHist(srcImgPatches[0], histsize));
    // printf("CSV loaded \n");
}

void patchMatcher_5(std::vector<cv::Mat> srcImgPatches,
                    std::vector<std::vector<float>>
                        &srcImgHists,
                    std::vector<char *> &imageNames,
                    std::vector<std::vector<std::vector<float>>> &features, cv::Mat filter,
                    const int histsize)
{


    /*
    Custom patch matcher that loads pre-computed S5S5 histogram features
    from a CSV file and computes the S5S5 histogram feature for each image patch,
    storing them in the provided data structures.
    */

    // lets us to add multiple feture vectors
    std::vector<std::vector<float>> featureVectors1, featureVectors2;
    std::vector<char *> dummy;
    read_image_data_csv("hm_s5s5_features.csv",
                        imageNames, featureVectors1, 0);
    // n is Number of images
    int n = featureVectors1.size();

    if (features.size() < n)
        features.resize(n);

    // Load the histogram feature vectors
    for (int i = 0; i < n; i++)
    {
        features[i].push_back(featureVectors1[i]);
    }
    srcImgHists.push_back(makeS5S5Hist(srcImgPatches[0], filter, histsize));
    // printf("CSV loaded \n");
}

int patchMatcher_6(char *srcFileName, std::vector<cv::Mat> srcImgPatches,
                   std::vector<std::vector<float>>
                       &srcImgHists,
                   std::vector<char *> &imageNames,
                   std::vector<std::vector<std::vector<float>>> &features,
                   const int histsize)
{

    /* Custom matchr which uses resnet features for matching. It loads pre-computed ResNet features
    from a CSV file and fetches the ResNet feature for the input image patch, storing them in the provided data structures.
    */
    std::vector<std::vector<float>> featureVectors1;

    std::vector<char *> dummy;

    read_image_data_csv("ResNet18_olym.csv",
                        dummy, featureVectors1, 0);
    imageNames = dummy;
    // printf("CSV loaded \n");
    // n is Number of images
    int n = featureVectors1.size();
    // H is number of histograms per image
    int H = 1;
    // D is number of bins in the histogram
    int D = featureVectors1[0].size();
    if (features.size() < n)
        features.resize(n);

    // Load the histogram feature vectors
    for (int i = 0; i < n; i++)
    {
        features[i].push_back(featureVectors1[i]);
    }

    // Used AI help to learn on how to use a custom comparator function in std binary search implementaion
    auto it = std::lower_bound(imageNames.begin(), imageNames.end(), srcFileName, [](char *srcName, char *targetName)
                               { return strcmp(srcName, targetName) < 0; });

    if (it != imageNames.end() && strcmp(*it, srcFileName) == 0)
    {
        int index = distance(imageNames.begin(), it);
        srcImgHists.push_back(featureVectors1[index]);
    }
    else
    {
        std::cout << "File name not found" << std::endl;
        return -1;
    }
    return 0;
}