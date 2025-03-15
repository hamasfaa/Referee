#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

// Lapangan
int hminLapangan = 70, sminLapangan = 186, vminLapangan = 146;
int hmaxLapangan = 122, smaxLapangan = 255, vmaxLapangan = 255;

// Team A
int hminTeamA = 72, sminTeamA = 13, vminTeamA = 73;
int hmaxTeamA = 126, smaxTeamA = 135, vmaxTeamA = 191;

// Team B
int hminTeamB = 59, sminTeamB = 66, vminTeamB = 0;
int hmaxTeamB = 255, smaxTeamB = 255, vmaxTeamB = 60;

// Ball
int hminBall = 0, sminBall = 162, vminBall = 125;
int hmaxBall = 51, smaxBall = 255, vmaxBall = 255;

// Line
int hminLine = 0, sminLine = 149, vminLine = 0;
int hmaxLine = 255, smaxLine = 255, vmaxLine = 255;

static const double thresholdAreaTimA = 150.0;
static const double thresholdAreaLapangan = 150.0;
static const double thresholdAreaBall = 150.0;
static const double toleranceDistance = 50.0;

enum BallStatus
{
    NONE,
    TEAM_A,
    TEAM_B
};

double calculateDistance(Point2f a, Point2f b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

double calculateMinDistance(Point2f point, Rect rect)
{
    Point2f rectPoints[4];
    rectPoints[0] = Point2f(rect.x, rect.y);
    rectPoints[1] = Point2f(rect.x + rect.width, rect.y);
    rectPoints[2] = Point2f(rect.x, rect.y + rect.height);
    rectPoints[3] = Point2f(rect.x + rect.width, rect.y + rect.height);

    double minDistance = calculateDistance(point, rectPoints[0]);
    for (int i = 1; i < 4; ++i)
    {
        double distance = calculateDistance(point, rectPoints[i]);
        if (distance < minDistance)
        {
            minDistance = distance;
        }
    }

    return minDistance;
}

int main(int, char **)
{
    string path = "../video/video3.mp4";
    VideoCapture cap(path);
    Mat frame, frameHSV, frameBlur, frameHSL;
    Mat maskLapangan, maskTeamA, kernel, maskBall, maskLine, maskTeamB;

    BallStatus lastBallStatus = NONE;

    // namedWindow("TrackbarsTeamB", (640, 200));
    // createTrackbar("Hue Min", "TrackbarsTeamB", &hminTeamB, 255);
    // createTrackbar("Hue Max", "TrackbarsTeamB", &hmaxTeamB, 255);
    // createTrackbar("Sat Min", "TrackbarsTeamB", &sminTeamB, 255);
    // createTrackbar("Sat Max", "TrackbarsTeamB", &smaxTeamB, 255);
    // createTrackbar("Val Min", "TrackbarsTeamB", &vminTeamB, 255);
    // createTrackbar("Val Max", "TrackbarsTeamB", &vmaxTeamB, 255);

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            cap >> frame;
        }

        cvtColor(frame, frameHSV, COLOR_BGR2HSV);
        cvtColor(frame, frameHSL, COLOR_BGR2HLS);
        blur(frameHSV, frameBlur, Size(3, 3));

        inRange(frameBlur, Scalar(hminLapangan, sminLapangan, vminLapangan), Scalar(hmaxLapangan, smaxLapangan, vmaxLapangan), maskLapangan);
        inRange(frameBlur, Scalar(hminTeamA, sminTeamA, vminTeamA), Scalar(hmaxTeamA, smaxTeamA, vmaxTeamA), maskTeamA);
        inRange(frameBlur, Scalar(hminTeamB, sminTeamB, vminTeamB), Scalar(hmaxTeamB, smaxTeamB, vmaxTeamB), maskTeamB);
        inRange(frameBlur, Scalar(hminBall, sminBall, vminBall), Scalar(hmaxBall, smaxBall, vmaxBall), maskBall);
        inRange(frameHSL, Scalar(hminLine, sminLine, vminLine), Scalar(hmaxLine, smaxLine, vmaxLine), maskLine);

        kernel = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
        morphologyEx(maskLapangan, maskLapangan, MORPH_CLOSE, kernel);

        vector<vector<Point>> contoursLapangan, contoursBall, contoursTeamA, contoursLine, contoursTeamB;
        vector<Vec4i> hierarchyLapangan, hierarchyBall, hierarchyTeamA, hierarchyLine, hierarchyTeamB;
        findContours(maskLapangan, contoursLapangan, hierarchyLapangan, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        findContours(maskTeamA, contoursTeamA, hierarchyTeamA, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        findContours(maskBall, contoursBall, hierarchyBall, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        findContours(maskTeamB, contoursTeamB, hierarchyTeamB, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Rect> boundingRects, boundingRectsTeamA, boundingRectsTeamB;
        Rect largestRectLapangan;
        int largestIndex = -1;
        double maxArea = 0;

        for (size_t i = 0; i < contoursLapangan.size(); i++)
        {
            double area = contourArea(contoursLapangan[i]);
            if (area > thresholdAreaLapangan)
            {
                Rect boundingRectLapangan = boundingRect(contoursLapangan[i]);
                bool merged = false;

                for (auto &rect : boundingRects)
                {
                    if ((abs(boundingRectLapangan.x - rect.x) < 1200) && (abs(boundingRectLapangan.y - rect.y) < 1200))
                    {
                        rect |= boundingRectLapangan;
                        merged = true;
                        break;
                    }
                }

                if (!merged)
                {
                    boundingRects.push_back(boundingRectLapangan);
                }

                if (area > maxArea)
                {
                    maxArea = area;
                    largestRectLapangan = boundingRectLapangan;
                    largestIndex = i;
                }
            }
        }

        for (const auto &rect : boundingRects)
        {
            rectangle(frame, rect, Scalar(0, 255, 0), 2);
        }

        for (size_t j = 0; j < contoursTeamA.size(); j++)
        {
            double areaTeamA = contourArea(contoursTeamA[j]);
            if (areaTeamA > thresholdAreaTimA)
            {
                Rect boundingRectTeamA = boundingRect(contoursTeamA[j]);
                if ((boundingRectTeamA & largestRectLapangan).area() > 0)
                {
                    bool merged = false;

                    for (auto &rect : boundingRectsTeamA)
                    {
                        rect |= boundingRectTeamA;
                        merged = true;
                    }

                    if (!merged)
                    {
                        boundingRectsTeamA.push_back(boundingRectTeamA);
                    }
                }
            }
        }

        for (const auto &rect : boundingRectsTeamA)
        {
            rectangle(frame, rect, Scalar(0, 0, 255), 2);
        }

        for (size_t j = 0; j < contoursTeamB.size(); j++)
        {
            double areaTeamB = contourArea(contoursTeamB[j]);
            if (areaTeamB > thresholdAreaTimA)
            {
                Rect boundingRectTeamB = boundingRect(contoursTeamB[j]);
                if ((boundingRectTeamB & largestRectLapangan).area() > 0)
                {
                    bool merged = false;

                    for (auto &rect : boundingRectsTeamB)
                    {
                        rect |= boundingRectTeamB;
                        merged = true;
                    }

                    if (!merged)
                    {
                        boundingRectsTeamB.push_back(boundingRectTeamB);
                    }
                }
            }
        }

        for (const auto &rect : boundingRectsTeamB)
        {
            rectangle(frame, rect, Scalar(255, 0, 0), 2);
        }

        for (size_t k = 0; k < contoursBall.size(); k++)
        {
            double areaBall = contourArea(contoursBall[k]);
            if (areaBall < thresholdAreaBall)
            {
                continue;
            }

            Point2f center;
            float radius;
            minEnclosingCircle(contoursBall[k], center, radius);
            bool isBallInsideLapangan = false;
            bool isBallInTeamA = false;
            bool isBallInTeamB = false;

            if (largestIndex != -1)
            {
                if (largestRectLapangan.contains(center))
                {
                    isBallInsideLapangan = true;
                }
            }

            for (const auto &rect : boundingRectsTeamA)
            {
                double distance = calculateMinDistance(center, rect);
                if (distance < radius + toleranceDistance)
                {
                    isBallInTeamA = true;
                    lastBallStatus = TEAM_A;
                    break;
                }
            }

            for (const auto &rect : boundingRectsTeamB)
            {
                double distance = calculateMinDistance(center, rect);
                if (distance < radius + toleranceDistance)
                {
                    isBallInTeamB = true;
                    lastBallStatus = TEAM_B;
                    break;
                }
            }

            circle(frame, center, (int)radius, Scalar(0, 255, 0), 2);
            if (isBallInsideLapangan)
            {
                putText(frame, "Bola di dalam lapangan", Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                if (lastBallStatus == TEAM_A)
                {
                    putText(frame, "Bola disentuh Tim A", Point(10, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                }
                else if (lastBallStatus == TEAM_B)
                {
                    putText(frame, "Bola disentuh Tim B", Point(10, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
                }
                else
                {
                    putText(frame, "Bola tidak disentuh Tim A/B", Point(10, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
                }
            }
            else
            {
                putText(frame, "Bola di luar lapangan", Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            }
        }

        imshow("Frame", frame);
        imshow("Mask Lapangan", maskLapangan);

        if (waitKey(18) == 27)
            break;
    }

    return 0;
}