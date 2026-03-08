/*
  Rohith Kumar Senthil Kumar
  Spring 2024
  CS 5330 Computer Vision

    A collection of image filters implemented as static functions for CBIR tasks.
*/

#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <random>

using std::string;
namespace cv2 = cv;

class Filters
{
public:
    static int sobelX3x3Gray(cv2::Mat &src, cv2::Mat &dst)
    {
        // Sobel X 3x3 Gray implementation using separable filters
        cv2::Mat dummy, srcGray;
        cv2::cvtColor(src, srcGray, cv2::COLOR_BGR2GRAY);
        dst = cv2::Mat::zeros(src.size(), CV_16SC1);
        dummy = cv2::Mat::zeros(src.size(), CV_16SC1);

        int rows = src.rows;
        int cols = src.cols;
        int gaussFilter[3] = {1, 2, 1};
        int diffFilter[3] = {-1, 0, 1};

        // Horizontal pass
        for (int i = 0; i < rows; i++)
        {
            uchar *src_row = srcGray.ptr<uchar>(i);
            short *dummy_row = dummy.ptr<short>(i);

            for (int j = 1; j < cols - 1; j++)
            {

                int sum = src_row[j - 1] * diffFilter[0] +
                          src_row[j] * diffFilter[1] +
                          src_row[j + 1] * diffFilter[2];
                dummy_row[j] = sum;
            }

            // Edge columns
            dummy_row[0] = src_row[0];
            dummy_row[cols - 1] = src_row[cols - 1];
        }

        // Vertical pass
        for (int i = 1; i < rows - 1; i++)
        {
            short *dst_row = dst.ptr<short>(i);
            short *dummy_row_m1 = dummy.ptr<short>(i - 1);
            short *dummy_row_0 = dummy.ptr<short>(i);
            short *dummy_row_p1 = dummy.ptr<short>(i + 1);

            for (int j = 0; j < cols; j++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int sum = dummy_row_m1[j] * gaussFilter[0] +
                              dummy_row_0[j] * gaussFilter[1] +
                              dummy_row_p1[j] * gaussFilter[2];
                    dst_row[j] = sum / 4;
                }
            }
        }

        // Edge rows
        dummy.row(0).copyTo(dst.row(0));
        dummy.row(rows - 1).copyTo(dst.row(rows - 1));

        return 0;
    }

    static int sobelY3x3Gray(cv2::Mat &src, cv2::Mat &dst)
    {

        // Sobel Y 3x3 Gray implementation using separable filters
        cv2::Mat dummy, srcGray;
        cv2::cvtColor(src, srcGray, cv2::COLOR_BGR2GRAY);
        dst = cv2::Mat::zeros(src.size(), CV_16SC1);
        dummy = cv2::Mat::zeros(src.size(), CV_16SC1);

        int rows = src.rows;
        int cols = src.cols;

        int gaussFilter[3] = {1, 2, 1};
        int diffFilter[3] = {-1, 0, 1};

        // Horizontal pass
        for (int i = 0; i < rows; i++)
        {
            uchar *src_row = srcGray.ptr<uchar>(i);
            short *dummy_row = dummy.ptr<short>(i);

            for (int j = 1; j < cols - 1; j++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int sum = src_row[j - 1] * gaussFilter[0] +
                              src_row[j] * gaussFilter[1] +
                              src_row[j + 1] * gaussFilter[2];
                    dummy_row[j] = sum / 4;
                }
            }

            // Edge columns

            dummy_row[0] = src_row[0];
            dummy_row[cols - 1] = src_row[cols - 1];
        }

        // Vertical pass
        for (int i = 1; i < rows - 1; i++)
        {
            short *dst_row = dst.ptr<short>(i);
            short *dummy_row_m1 = dummy.ptr<short>(i - 1);
            short *dummy_row_0 = dummy.ptr<short>(i);
            short *dummy_row_p1 = dummy.ptr<short>(i + 1);

            for (int j = 0; j < cols; j++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int sum = dummy_row_m1[j] * diffFilter[0] +
                              dummy_row_0[j] * diffFilter[1] +
                              dummy_row_p1[j] * diffFilter[2];
                    dst_row[j] = sum;
                }
            }
        }

        // Edge rows

        dummy.row(0).copyTo(dst.row(0));
        dummy.row(rows - 1).copyTo(dst.row(rows - 1));

        return 0;
    }

    static int magnitude(cv2::Mat &sx, cv2::Mat &sy, cv2::Mat &dst)
    {
        // Magnitude calculation from sobel x and y
        int rows = sx.rows;
        int cols = sx.cols;
        dst = cv2::Mat::zeros(sx.size(), CV_16SC1);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // sobel x value in the pixel
                double sxv = sx.at<short>(i, j);
                // sobel y value in the pixel
                double syv = sy.at<short>(i, j);
                dst.at<short>(i, j) = static_cast<short>(std::sqrt((sxv * sxv) + (syv * syv)));
            }
        }
        return 0;
    }

    static int orientation(cv2::Mat &sx, cv2::Mat &sy, cv2::Mat &dst)
    {
        // Orientation calculation from sobel x and y
        int rows = sx.rows;
        int cols = sx.cols;
        dst = cv2::Mat::zeros(sx.size(), CV_32FC1);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // sobel x value in the pixel
                double sxv = sx.at<short>(i, j);
                // sobel y value in the pixel
                double syv = sy.at<short>(i, j);
                dst.at<float>(i, j) = static_cast<float>(std::atan2(syv, sxv));
            }
        }
        return 0;
    }

    static void contrast(cv2::Mat &image, cv2::Mat &output, double contrast)
    {
        // Contrast and offset parameters
        // >1 increases contrast, <1 decreases
        int offset = 30; // Brightness offset

        // Create output image
        output = cv2::Mat::zeros(image.size(), image.type());

        // Apply contrast adjustment: output = contrast * input + offset
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
            {
                for (int c = 0; c < image.channels(); c++)
                {
                    int value = contrast * image.at<cv2::Vec3b>(y, x)[c] + offset;
                    // Clamp to valid range [0, 255]
                    output.at<cv2::Vec3b>(y, x)[c] = cv2::saturate_cast<uchar>(value);
                }
            }
        }
    }
};