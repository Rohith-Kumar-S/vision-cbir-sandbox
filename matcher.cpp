/*
  Rohith Kumar Senthil Kumar
  Spring 2024
  CS 5330

  image matching system using histogram features, texture features, and DNN features.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include <opencv2/core/utils/logger.hpp>
#include "cbir_utils.h"

int main(int argc, char *argv[])
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // Initialize variables
    cv::Mat src;
    cv::Mat dst;
    cv::Mat filter;

    char filename[256];
    int topNCount = 3;
    // color feature type
    char matcher1[14] = "rg";
    // texture feature type
    char matcher2[14] = "None";
    // dnn feature type
    char matcher3[14] = "None";
    char matchDirection[7] = "top";
    char distanceMetric[7] = "l1";
    float alpha = 0.0f;
    float beta = 0.0f;
    float gamma = 0.0f;
    int colorPatchType = 0;
    int texturePatchType = 0;
    int dnnPatchType = 0;
    const int histsize = 16;

    // error checking
    if (argc < 2)
    {
        printf("usage: %s <image filename>\n", argv[0]);
        return (-1);
    }

    // parse command line arguments
    if (argc >= 10)
    {
        gamma = std::stof(argv[9]);
    }
    if (argc >= 9)
    {
        beta = std::stof(argv[8]);
    }
    if (argc >= 8)
    {
        alpha = std::stof(argv[7]);
    }
    if (argc >= 7)
    {
        strcpy(matcher3, argv[6]);
    }
    if (argc >= 6)
    {
        strcpy(matcher2, argv[5]);
        if (strcmp(matcher2, "texture-grads") == 0)
            texturePatchType = 0;
        else if (strcmp(matcher2, "texture-s5s5") == 0)
            texturePatchType = 1;
    }
    if (argc >= 5)
    {
        strcpy(matcher1, argv[4]);
        if (strcmp(matcher1, "center-patch") == 0)
            colorPatchType = 0;
        else if (strcmp(matcher1, "rg") == 0)
            colorPatchType = 1;
        else if (strcmp(matcher1, "rgb") == 0)
            colorPatchType = 2;
        else if (strcmp(matcher1, "multi-rg") == 0)
            colorPatchType = 3;
    }

    if (argc >= 4)
    {
        topNCount = std::atoi(argv[3]);
        // As I am skipping first match as its the source image itself.
        topNCount += 1;
    }

    if (argc >= 3)
        strcpy(matchDirection, argv[2]);

    // grab the filename
    strcpy(filename, argv[1]);

    // read the file
    src = cv::imread(filename);
    if (src.data == NULL)
    {
        printf("error: unable to read filename %s\n", filename);
        return (-2);
    }

    // More initialization
    std::vector<char *> imageNames;
    std::vector<char *> dummy;
    std::vector<std::vector<std::vector<float>>> targetFeatures;
    std::vector<std::vector<float>> srcImgHists;
    std::vector<cv::Mat> srcImgPatches;
    std::vector<float> distances;
    std::priority_queue<std::pair<float, std::string>,
                        std::vector<std::pair<float, std::string>>,
                        std::greater<std::pair<float, std::string>>>
        minHeap;
    std::priority_queue<std::pair<float, std::string>> maxHeap;
    std::vector<std::string> topNMatches;

    // Initially setting feature count map to 1 for all features,
    // as we are dividing the weightage by the count of features to ensure
    // that the total weightage for each feature type remains consistent even
    // if there are multiple features of the same type included.
    std::unordered_map<int, int> featureCountMap = {{0, 1}, {1, 1}, {2, 1}};
    std::vector<int> featuresIncluded;

    // Compute S5S5 filter for texture feature extraction
    computeS5S5LawsFilter(filter);

    // If color features are included, compute color features and load corresponding histogram features from CSV.
    if (std::strcmp(matcher1, "None") != 0)
    {
        // using intersection as distance metric for color feature type except for center patch
        strcpy(distanceMetric, "inter");
        if (strcmp(matcher1, "center-patch") == 0)
            strcpy(distanceMetric, "l2");
        switch (colorPatchType)
        {
        case 0:
        {
            // center patch features
            srcImgPatches.push_back(src);
            featuresIncluded.push_back(0);
            featureCountMap[0] = 1;
            patchMatcher_0(srcImgPatches, srcImgHists, imageNames, targetFeatures);
            break;
        }
        case 1:
        {
            // rg histogram features
            srcImgPatches.push_back(src);
            featuresIncluded.push_back(0);
            featureCountMap[0] = 1;
            patchMatcher_1(srcImgPatches, srcImgHists, imageNames, targetFeatures, histsize);
            break;
        }
        case 2:
        {
            // rgb histogram features
            srcImgPatches.push_back(src);
            featuresIncluded.push_back(0);
            featureCountMap[0] = 1;
            patchMatcher_2(srcImgPatches, srcImgHists, imageNames, targetFeatures, 8);
            break;
        }
        case 3:
        {
            // multi-rg histogram features (top half and bottom half)
            srcImgPatches.push_back(src(cv::Rect(0, 0, src.cols, src.rows / 2)));
            srcImgPatches.push_back(src(cv::Rect(0, src.rows / 2, src.cols, src.rows / 2)));
            featuresIncluded.push_back(0);
            featuresIncluded.push_back(0);
            featureCountMap[0] = 2;
            patchMatcher_3(srcImgPatches, srcImgHists, imageNames, targetFeatures, histsize);
            break;
        }

        default:
            break;
        }
    }

    // If texture features are included, compute texture features and load corresponding histogram features from CSV.
    if (std::strcmp(matcher2, "None") != 0)
    {
        // using l1 distance metric for texture features
        strcpy(distanceMetric, "l1");
        switch (texturePatchType)
        {
        case 0:
        {
            // texture gradient histogram features using gradient magnitudes and orientations
            srcImgPatches.push_back(src);
            if (std::strcmp(matcher1, "None") != 0)
                patchMatcher_4(srcImgPatches, srcImgHists, dummy, targetFeatures, histsize);
            else
                patchMatcher_4(srcImgPatches, srcImgHists, imageNames, targetFeatures, histsize);
            featuresIncluded.push_back(1);
            featuresIncluded.push_back(1);
            featureCountMap[1] = 2;
            break;
        }
        case 1:
        {
            // texture histogram features using Laws' texture energy measures (S5S5 filter)
            srcImgPatches.push_back(src);
            if (std::strcmp(matcher1, "None") != 0)
                patchMatcher_5(srcImgPatches, srcImgHists, dummy, targetFeatures, filter, histsize);
            else
                patchMatcher_5(srcImgPatches, srcImgHists, imageNames, targetFeatures, filter, histsize);
            featuresIncluded.push_back(1);
            featureCountMap[1] = 2;
            break;
        }

        default:
            break;
        }
    }

    // If DNN features are included, load DNN features from CSV.
    if (std::strcmp(matcher3, "None") != 0)
    {
        strcpy(distanceMetric, "cosine");
        switch (dnnPatchType)
        {
        case 0:
        {
            // DNN features using cosine distance metric
            char *fileNameExtract = strrchr(filename, '\\');
            if (fileNameExtract != nullptr)
                fileNameExtract++;
            else
            {
                fileNameExtract = strrchr(filename, '/');
                if (fileNameExtract != nullptr)
                    fileNameExtract++;
                else
                    fileNameExtract = filename;
            }
            srcImgPatches.push_back(src);
            int res = -1;
            if (std::strcmp(matcher1, "None") != 0 || std::strcmp(matcher2, "None") != 0)
            {
                res = patchMatcher_6(fileNameExtract, srcImgPatches, srcImgHists, dummy, targetFeatures, histsize);
            }
            else
                res = patchMatcher_6(fileNameExtract, srcImgPatches, srcImgHists, imageNames, targetFeatures, histsize);
            if (res == -1)
                imageNames = {};
            featuresIncluded.push_back(2);
            featureCountMap[2] = 1;
            break;
        }

        default:
            break;
        }
    }
    //

    bool topMatches = (strcmp(matchDirection, "top") == 0);

    // custom weightage is applied when alpha, beta, or gamma values are provided for color, texture, and DNN features respectively.
    bool customWeightage = alpha != 0.0f || beta != 0.0f || gamma != 0.0f;

    // Compute distances between source image features and target image features, and store in min-heap or max-heap based on match direction.
    for (int imgId = 0; imgId < imageNames.size(); imgId++)
    {
        float distance = 0.0f;

        if (customWeightage)
        {
            int numFeatures = targetFeatures[imgId].size();
            for (int f = 0; f < numFeatures; ++f)
            {
                float weight = 0.0f;
                std::string metric;

                if (featuresIncluded[f] == 0)
                {
                    weight = alpha / std::max(1, featureCountMap[0]);
                    metric = "inter";
                }
                else if (featuresIncluded[f] == 1)
                {
                    weight = beta / featureCountMap[1];
                    metric = "l1";
                }
                else if (featuresIncluded[f] == 2)
                {
                    weight = gamma / featureCountMap[2];
                    metric = "cosine";
                }

                // If weight is zero, skip the distance computation for this feature.
                if (weight == 0.0f)
                    continue;
                
                // Compute distance for this feature and accumulate weighted distance.
                float d = computeDistance(srcImgHists[f],
                                          targetFeatures[imgId][f],
                                          metric);
                distance += weight * d;
            }
        }
        else
        {
            // If no custom weightage is provided, compute simple average distance across all features.
            for (int histId = 0; histId < targetFeatures[0].size(); histId++)
            {
                distance += computeDistance(srcImgHists[histId], targetFeatures[imgId][histId], distanceMetric);
            }
            distance /= srcImgHists.size();
        }
        // Store distance in min-heap for top matches and max-heap for bottom matches.
        if (topMatches)
            minHeap.push({distance, imageNames[imgId]});
        else
            maxHeap.push({distance, imageNames[imgId]});
    }

    // Retrieve and print top N matches from the appropriate heap based on match direction.
    for (int i = 0; i < topNCount && (!minHeap.empty() || !maxHeap.empty()); i++)
    {
        if (topMatches)
        {
            if (i == 0)
            {
                minHeap.pop();
                continue;
            }
            std::cout << minHeap.top().second << " | Distance: " << minHeap.top().first << std::endl;
            minHeap.pop();
        }
        else
        {
            if (i == 0)
            {
                maxHeap.pop();
                continue;
            }
            std::cout << maxHeap.top().second << " | Distance: " << maxHeap.top().first << std::endl;
            maxHeap.pop();
        }
    }

    return 0;
}